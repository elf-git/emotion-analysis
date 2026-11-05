"""
train_video.py - Video Emotion Recognition Training Pipeline
Prioritizes pre-extracted visual features for initial baselines, with full support
for raw video clips (ResNet + Temporal Attention) when present and needed.
Includes checkpoints, GPU memory controls, mixed precision (AMP), and complete metrics.
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
from ml.data.media_assets import load_video_features, inspect_media_assets, extract_frames_from_clip
from ml.models.video_model import VideoFeatureModel, VideoFrameModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MELDVideoFeatureDataset(Dataset):
    """MELD dataset for visual emotion classification using pre-extracted features."""

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

        logger.info(f"Loaded {len(self.features)} visual utterances (dim={feature_dim})")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "video_features": self.features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "dialogue_id": self.dialogue_ids[idx],
            "utterance_id": self.utterance_ids[idx],
        }


class MELDVideoClipDataset(Dataset):
    """MELD dataset loading raw MP4 video clips and sampling frames."""

    EMOTION_MAP = {
        "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
        "joy": 4, "disgust": 5, "anger": 6,
    }

    def __init__(self, utterances: List[Dict], video_root: str, split: str, num_frames: int = 8):
        self.num_frames = num_frames
        self.items = []

        for u in utterances:
            clip_name = f"dia{u['dialogue_id']}_utt{u['utterance_id']}.mp4"
            # Support either split subdirectory or flat directory
            candidates = [
                os.path.join(video_root, split, clip_name),
                os.path.join(video_root, clip_name),
            ]
            clip_path = next((c for c in candidates if os.path.exists(c)), None)
            if clip_path:
                self.items.append({
                    "path": clip_path,
                    "label": self.EMOTION_MAP.get(u["emotion"], 0),
                    "dialogue_id": u["dialogue_id"],
                    "utterance_id": u["utterance_id"],
                })

        logger.info(f"Loaded {len(self.items)} available raw clips for {split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = extract_frames_from_clip(item["path"], num_frames=self.num_frames)
        return {
            "frames": frames,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "dialogue_id": item["dialogue_id"],
            "utterance_id": item["utterance_id"],
        }


def train_one_epoch_features(
    model, dataloader, optimizer, criterion, device, scheduler=None,
    max_grad_norm: float = 1.0, grad_accum_steps: int = 1,
):
    """Train for one epoch using pre-extracted visual features."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        feats = batch["video_features"].to(device)
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
def evaluate_features(model, dataloader, criterion, device):
    """Evaluate visual feature model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    use_amp = (device.type == "cuda")
    for batch in dataloader:
        feats = batch["video_features"].to(device)
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


def train_video_model(
    config: Dict,
    smoke_test: bool = False,
    resume_from: Optional[str] = None,
    use_raw_clips: bool = False,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
):
    """
    Main training execution function for video emotion recognition.
    Prioritizes pre-extracted visual features for initial baselines; uses raw video clips when requested.
    """
    set_seed(config["training"]["seed"])
    device = get_device(config["kaggle"]["gpu_memory_fraction"])
    paths = get_paths(config)
    data_dir = paths.get("data_dir", "ml/data/raw")

    # 1. Load CSV Splits
    splits = prepare_dataset(data_dir)

    # 2. Inspect Assets
    assets = inspect_media_assets(data_dir)
    has_raw_clips = assets["raw_video_dir"] is not None and sum(assets["clip_counts"].values()) > 0
    run_raw_clips = use_raw_clips and has_raw_clips

    logger.info(
        f"Media Assets Audit: Raw clips available: {has_raw_clips} (counts={assets['clip_counts']}), "
        f"Pipeline mode: {'Raw Frame ResNet' if run_raw_clips else 'Pre-extracted Visual Features'}"
    )

    feature_dim = config["model"]["video"].get("feature_dim", 512)

    if smoke_test:
        splits = create_smoke_test_subset(splits, config["training"]["smoke_test_samples"])
        epochs = epochs_override or config["training"]["epochs_smoke"]
        batch_size = batch_size_override or config["training"]["batch_size_smoke"]
        logger.info("*** VIDEO SMOKE TEST MODE ***")
    else:
        epochs = epochs_override or config["training"]["epochs"]
        batch_size = batch_size_override or config["training"]["batch_size"]
        num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
        if num_gpus > 1 and batch_size_override is None:
            batch_size = batch_size * num_gpus
            logger.info(f"Scaled batch size to {batch_size} across {num_gpus} GPUs.")

    num_workers = 4 if is_kaggle() else (config["kaggle"]["num_workers"] if is_kaggle() else 0)
    pin_mem = config["kaggle"]["pin_memory"] if device.type == "cuda" else False

    # 3. Model & DataLoader Branching (Feature Baseline vs Raw Frame Model)
    if run_raw_clips:
        logger.info("Initializing Raw Frame Video Pipeline with ResNet-18 + Temporal Attention...")
        train_ds = MELDVideoClipDataset(splits["train"], assets["raw_video_dir"], "train")
        val_ds = MELDVideoClipDataset(splits["dev"], assets["raw_video_dir"], "dev")
        test_ds = MELDVideoClipDataset(splits["test"], assets["raw_video_dir"], "test")

        # Memory control: Adjust batch size for raw video frames across GPUs to prevent OOM
        frame_batch_size = max(4, batch_size // 2)
        train_dl = DataLoader(train_ds, batch_size=frame_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        val_dl = DataLoader(val_ds, batch_size=frame_batch_size, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, batch_size=frame_batch_size, shuffle=False, num_workers=num_workers)

        model = VideoFrameModel(
            backbone=config["model"]["video"].get("backbone", "resnet18"),
            num_classes=config["model"]["fusion"]["num_classes_emotion"],
            hidden_dim=config["model"]["video"]["hidden_dim"],
            dropout=config["model"]["video"]["dropout"],
            frames_per_utterance=config["model"]["video"].get("frames_per_utterance", 8),
        )
        model, is_multigpu = wrap_model_multigpu(model, device)
    else:
        # Pre-extracted Visual Feature Baseline
        logger.info("Initializing Pre-extracted Visual Feature Baseline...")
        train_feats, dev_feats, test_feats, feature_dim = load_video_features(
            data_dir, splits, feature_dim=feature_dim, seed=config["training"]["seed"]
        )
        train_ds = MELDVideoFeatureDataset(splits["train"], train_feats, feature_dim)
        val_ds = MELDVideoFeatureDataset(splits["dev"], dev_feats, feature_dim)
        test_ds = MELDVideoFeatureDataset(splits["test"], test_feats, feature_dim)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

        model = VideoFeatureModel(
            input_dim=feature_dim,
            hidden_dim=config["model"]["video"]["hidden_dim"],
            num_classes=config["model"]["fusion"]["num_classes_emotion"],
            dropout=config["model"]["video"]["dropout"],
            proj_dim=config["model"]["fusion"]["hidden_dim"],
        )
        model, is_multigpu = wrap_model_multigpu(model, device)

    # 4. Class Weights for Imbalanced Loss
    emotion_counts = {}
    for u in splits["train"]:
        e = u["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    class_weights = get_class_weights(emotion_counts, MELDVideoFeatureDataset.EMOTION_MAP).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3 if not run_raw_clips else 2e-4,
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5. Checkpoints & Resumption Management
    ckpt_dir = paths.get("checkpoint_dir") or config.get("training", {}).get("checkpoint_dir", "checkpoints")
    if not is_kaggle() and (ckpt_dir.startswith("/kaggle") or not os.path.isabs(ckpt_dir)):
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_mgr = CheckpointManager(ckpt_dir, "video_model")
    early_stop = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    start_epoch = 0
    if resume_from:
        ckpt = ckpt_mgr.load(model, optimizer, scheduler, path=resume_from)
        if ckpt:
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resuming video model training from epoch {start_epoch+1}")

    # 6. Training Loop
    history = []
    for epoch in range(start_epoch, epochs):
        logger.info(f"\n--- Video Epoch {epoch+1}/{epochs} ---")

        train_metrics = train_one_epoch_features(
            model, train_dl, optimizer, criterion, device, scheduler,
            config["training"]["max_grad_norm"],
            config["training"]["gradient_accumulation_steps"],
        )
        val_metrics = evaluate_features(model, val_dl, criterion, device)

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

        # GPU memory control: Periodic cache clearance to avoid fragmentation
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if early_stop.step(val_metrics["f1_weighted"]):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 7. Final Test Evaluation
    ckpt_mgr.load(model)
    test_metrics = evaluate_features(model, test_dl, criterion, device)
    logger.info(f"\nVideo Final Test Results: Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_weighted']:.4f}")

    results = {
        "model": "video",
        "mode": "raw_frames" if run_raw_clips else "pre_extracted_features",
        "feature_dim": feature_dim,
        "smoke_test": smoke_test,
        "history": history,
        "test_metrics": test_metrics,
    }
    results_path = os.path.join(ckpt_dir, "video_model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Video results saved to {results_path}")

    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train video emotion recognition model on MELD")
    parser.add_argument("--config", default="configs/config.json", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test")
    parser.add_argument("--use-raw-clips", action="store_true", help="Use raw video clips if available")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_video_model(
        config,
        smoke_test=args.smoke_test,
        resume_from=args.resume,
        use_raw_clips=args.use_raw_clips,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )
