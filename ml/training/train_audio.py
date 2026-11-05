"""
train_audio.py - Audio Emotion Recognition Training Pipeline
Prioritizes pre-extracted openSMILE audio features for fast baseline training.
Includes checkpoints, GPU memory controls, mixed precision (AMP), early stopping, and complete evaluation metrics.
"""
import os
import sys
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.training.train_utils import (
    get_device, set_seed, CheckpointManager, EarlyStopping,
    compute_metrics, get_class_weights, get_paths, is_kaggle,
    wrap_model_multigpu,
)
from ml.data.prepare_meld import prepare_dataset, create_smoke_test_subset
from ml.data.media_assets import load_audio_features, inspect_media_assets
from ml.models.audio_model import AudioFeatureModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MELDAudioDataset(Dataset):
    """MELD dataset for audio emotion classification using pre-extracted features."""

    EMOTION_MAP = {
        "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
        "joy": 4, "disgust": 5, "anger": 6,
    }

    def __init__(self, utterances: List[Dict], features_dict: Dict[str, Any], feature_dim: int):
        self.features = []
        self.labels = []
        self.dialogue_ids = []
        self.utterance_ids = []

        default_feat = torch.zeros(feature_dim, dtype=torch.float32)

        for u in utterances:
            key = f"{u['dialogue_id']}_{u['utterance_id']}"
            if key in features_dict:
                feat = torch.tensor(features_dict[key], dtype=torch.float32)
            else:
                feat = default_feat

            self.features.append(feat)
            self.labels.append(self.EMOTION_MAP.get(u["emotion"], 0))
            self.dialogue_ids.append(u["dialogue_id"])
            self.utterance_ids.append(u["utterance_id"])

        logger.info(f"Loaded {len(self.features)} audio utterances (dim={feature_dim})")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "audio_features": self.features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "dialogue_id": self.dialogue_ids[idx],
            "utterance_id": self.utterance_ids[idx],
        }


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, scheduler=None,
    max_grad_norm: float = 1.0, grad_accum_steps: int = 1,
):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        feats = batch["audio_features"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(feats)
            loss = criterion(outputs["logits"], labels) / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        preds = outputs["logits"].argmax(dim=-1).detach().cpu()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels, num_classes=7)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate audio model on validation or test split."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    use_amp = (device.type == "cuda")
    for batch in dataloader:
        feats = batch["audio_features"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(feats)
            loss = criterion(outputs["logits"], labels)

        total_loss += loss.item()
        preds = outputs["logits"].argmax(dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels, num_classes=7)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def train_audio_model(
    config: Dict,
    smoke_test: bool = False,
    resume_from: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
):
    """
    Main training execution function for audio emotion classification.
    """
    set_seed(config["training"]["seed"])
    device = get_device(config["kaggle"]["gpu_memory_fraction"])
    paths = get_paths(config)
    data_dir = paths.get("data_dir", "ml/data/raw")

    # 1. Load CSV Splits
    splits = prepare_dataset(data_dir)

    # 2. Inspect Assets & Load Pre-extracted Audio Features
    assets = inspect_media_assets(data_dir)
    logger.info(
        f"Media Assets Audit: Raw tarball={'Found' if assets['raw_tarball'] else 'None'}, "
        f"Features tarball={'Found' if assets['features_tarball'] else 'None'}, "
        f"Audio pickles={len(assets['audio_pickles'])}, Raw clips={sum(assets['clip_counts'].values())}"
    )

    feature_dim = config["model"]["audio"].get("feature_dim", 1611)
    train_feats, dev_feats, test_feats, feature_dim = load_audio_features(
        data_dir, splits, feature_dim=feature_dim, seed=config["training"]["seed"]
    )

    if smoke_test:
        splits = create_smoke_test_subset(splits, config["training"]["smoke_test_samples"])
        epochs = epochs_override or config["training"]["epochs_smoke"]
        batch_size = batch_size_override or config["training"]["batch_size_smoke"]
        logger.info("*** AUDIO SMOKE TEST MODE ***")
    else:
        epochs = epochs_override or config["training"]["epochs"]
        batch_size = batch_size_override or config["training"]["batch_size"]
        num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
        if num_gpus > 1 and batch_size_override is None:
            batch_size = batch_size * num_gpus
            logger.info(f"Scaled batch size to {batch_size} across {num_gpus} GPUs.")

    # 3. Datasets and Loaders
    train_ds = MELDAudioDataset(splits["train"], train_feats, feature_dim)
    val_ds = MELDAudioDataset(splits["dev"], dev_feats, feature_dim)
    test_ds = MELDAudioDataset(splits["test"], test_feats, feature_dim)

    num_workers = 4 if is_kaggle() else (config["kaggle"]["num_workers"] if is_kaggle() else 0)
    pin_mem = config["kaggle"]["pin_memory"] if device.type == "cuda" else False

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    # 4. Model Setup
    model = AudioFeatureModel(
        input_dim=feature_dim,
        hidden_dim=config["model"]["audio"]["hidden_dim"],
        num_classes=config["model"]["fusion"]["num_classes_emotion"],
        dropout=config["model"]["audio"]["dropout"],
        proj_dim=config["model"]["fusion"]["hidden_dim"],
    )
    model, is_multigpu = wrap_model_multigpu(model, device)

    # 5. Class Weights for Imbalanced Loss
    emotion_counts = {}
    for u in splits["train"]:
        e = u["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    class_weights = get_class_weights(emotion_counts, MELDAudioDataset.EMOTION_MAP).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 6. Checkpoint & Resumption Management
    ckpt_dir = paths.get("checkpoint_dir") or config.get("training", {}).get("checkpoint_dir", "checkpoints")
    if not is_kaggle() and (ckpt_dir.startswith("/kaggle") or not os.path.isabs(ckpt_dir)):
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_mgr = CheckpointManager(ckpt_dir, "audio_model")
    early_stop = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    start_epoch = 0
    if resume_from:
        ckpt = ckpt_mgr.load(model, optimizer, scheduler, path=resume_from)
        if ckpt:
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resuming audio model training from epoch {start_epoch+1}")

    # 7. Training Loop
    history = []
    for epoch in range(start_epoch, epochs):
        logger.info(f"\n--- Audio Epoch {epoch+1}/{epochs} ---")

        train_metrics = train_one_epoch(
            model, train_dl, optimizer, criterion, device, scheduler,
            config["training"]["max_grad_norm"],
            config["training"]["gradient_accumulation_steps"],
        )
        val_metrics = evaluate(model, val_dl, criterion, device)

        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1_weighted']:.4f}"
        )
        logger.info(
            f"Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_weighted']:.4f}"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        ckpt_mgr.save(
            model, optimizer, epoch,
            {"val_f1_weighted": val_metrics["f1_weighted"], **val_metrics},
            scheduler,
        )

        if early_stop.step(val_metrics["f1_weighted"]):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 8. Final Test Set Evaluation
    ckpt_mgr.load(model)
    test_metrics = evaluate(model, test_dl, criterion, device)
    logger.info(f"\nAudio Final Test Results: Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_weighted']:.4f}")

    results = {
        "model": "audio",
        "feature_dim": feature_dim,
        "smoke_test": smoke_test,
        "history": history,
        "test_metrics": test_metrics,
    }
    results_path = os.path.join(ckpt_dir, "audio_model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Audio results saved to {results_path}")

    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train audio emotion recognition model on MELD")
    parser.add_argument("--config", default="configs/config.json", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_audio_model(
        config,
        smoke_test=args.smoke_test,
        resume_from=args.resume,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )
