"""
train_text.py - Text Model Training Script
Trains BERT-based text emotion classifier on MELD utterances.
Designed for Kaggle GPU runtime with checkpoint resume support.
"""
import os
import sys
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.training.train_utils import (
    get_device, set_seed, CheckpointManager, EarlyStopping,
    compute_metrics, get_class_weights, is_kaggle, get_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MELDTextDataset(Dataset):
    """MELD dataset for text-only emotion recognition."""

    EMOTION_MAP = {
        "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
        "joy": 4, "disgust": 5, "anger": 6,
    }

    def __init__(self, utterances: List[Dict], tokenizer, max_length: int = 128):
        self.utterances = utterances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        item = self.utterances[idx]
        text = item["utterance"]
        label = self.EMOTION_MAP.get(item["emotion"], 0)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "dialogue_id": item["dialogue_id"],
            "utterance_id": item["utterance_id"],
        }


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, scheduler=None,
    max_grad_norm: float = 1.0, grad_accum_steps: int = 1,
):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
        preds = outputs["logits"].argmax(dim=-1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels, num_classes=7)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset split."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs["logits"], labels)

        total_loss += loss.item()
        preds = outputs["logits"].argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels, num_classes=7)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def train_text_model(
    config: Dict,
    smoke_test: bool = False,
    resume_from: Optional[str] = None,
    backbone_override: Optional[str] = None,
):
    """
    Full training pipeline for text emotion model.

    Args:
        config: Configuration dictionary
        smoke_test: If True, use small subset and fewer epochs
        resume_from: Path to checkpoint to resume from
        backbone_override: Optional override for pretrained backbone
    """
    from transformers import AutoTokenizer
    from ml.models.text_model import TextEmotionModel
    from ml.data.prepare_meld import prepare_dataset, create_smoke_test_subset

    set_seed(config["training"]["seed"])
    device = get_device(config["kaggle"]["gpu_memory_fraction"])

    # Load data
    paths = get_paths(config)
    data_dir = paths.get("data_dir", "ml/data/raw")
    splits = prepare_dataset(data_dir)

    if smoke_test:
        splits = create_smoke_test_subset(splits, config["training"]["smoke_test_samples"])
        epochs = config["training"]["epochs_smoke"]
        batch_size = config["training"]["batch_size_smoke"]
        logger.info("*** SMOKE TEST MODE ***")
    else:
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]

    # Tokenizer and datasets
    backbone = backbone_override or config["model"]["text"]["backbone"]
    if smoke_test and backbone_override is None:
        # Default to lightweight distilbert for smoke test
        backbone = "distilbert-base-uncased"
    logger.info(f"Using pretrained backbone: {backbone}")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    max_len = config["model"]["text"]["max_seq_length"]

    train_ds = MELDTextDataset(splits["train"], tokenizer, max_len)
    val_ds = MELDTextDataset(splits["dev"], tokenizer, max_len)
    test_ds = MELDTextDataset(splits["test"], tokenizer, max_len)

    num_workers = config["kaggle"]["num_workers"] if is_kaggle() else 0
    pin_mem = config["kaggle"]["pin_memory"] if device.type == "cuda" else False

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers,
                          pin_memory=pin_mem)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                        num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                         num_workers=num_workers)

    # Model
    model = TextEmotionModel(
        backbone=backbone,
        num_classes=config["model"]["fusion"]["num_classes_emotion"],
        hidden_dim=config["model"]["text"]["hidden_dim"],
        dropout=config["model"]["text"]["dropout"],
        freeze_layers=config["model"]["text"]["freeze_layers"],
    ).to(device)

    # Class weights for imbalanced data
    emotion_counts = {}
    for u in splits["train"]:
        e = u["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    class_weights = get_class_weights(emotion_counts, MELDTextDataset.EMOTION_MAP).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Checkpoint manager
    ckpt_dir = paths.get("checkpoint_dir") or config.get("training", {}).get("checkpoint_dir", "checkpoints")
    if not is_kaggle() and (ckpt_dir.startswith("/kaggle") or not os.path.isabs(ckpt_dir)):
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_mgr = CheckpointManager(ckpt_dir, "text_model")
    early_stop = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    start_epoch = 0
    if resume_from:
        ckpt = ckpt_mgr.load(model, optimizer, scheduler, resume_from)
        if ckpt:
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    history = []
    for epoch in range(start_epoch, epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")

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

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        ckpt_mgr.save(
            model, optimizer, epoch,
            {"val_f1_weighted": val_metrics["f1_weighted"], **val_metrics},
            scheduler,
        )

        if early_stop.step(val_metrics["f1_weighted"]):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final test evaluation
    ckpt_mgr.load(model)  # Load best model
    test_metrics = evaluate(model, test_dl, criterion, device)
    logger.info(f"\nTest Results: Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_weighted']:.4f}")

    # Save results
    results = {
        "model": "text",
        "backbone": backbone,
        "smoke_test": smoke_test,
        "history": history,
        "test_metrics": test_metrics,
    }
    results_path = os.path.join(ckpt_dir, "text_model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train text emotion classification model on MELD")
    parser.add_argument("--config", default="configs/config.json", help="Path to config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--backbone", default=None, help="Pretrained model backbone override")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.epochs is not None:
        if args.smoke_test:
            config["training"]["epochs_smoke"] = args.epochs
        else:
            config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        if args.smoke_test:
            config["training"]["batch_size_smoke"] = args.batch_size
        else:
            config["training"]["batch_size"] = args.batch_size

    train_text_model(
        config,
        smoke_test=args.smoke_test,
        resume_from=args.resume,
        backbone_override=args.backbone,
    )
