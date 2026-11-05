"""
train_utils.py - Shared training utilities for all modalities.
Provides trainer class, checkpoint management, metrics, and Kaggle compatibility.
"""
import os
import json
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def get_device(gpu_memory_fraction: float = 0.9) -> torch.device:
    """Get the primary device and log all available GPU devices."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU [{i}]: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
        return device
    else:
        logger.info("No GPU available, using CPU")
        return torch.device("cpu")


def wrap_model_multigpu(model: nn.Module, device: torch.device) -> Tuple[nn.Module, bool]:
    """
    Wrap model with nn.DataParallel if multiple GPUs are available.
    Returns (model, is_multigpu).
    """
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Enabling DataParallel across {n_gpus} GPUs.")
        model = nn.DataParallel(model)
        return model, True
    return model, False


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CheckpointManager:
    """Save and load model checkpoints with Kaggle-compatible paths and DataParallel unwrapping."""

    def __init__(self, checkpoint_dir: str, model_name: str, max_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.max_keep = max_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        scheduler=None,
        extra: Optional[Dict] = None,
    ) -> str:
        """Save a checkpoint and return its path (unwrapping DataParallel if active)."""
        # Always unwrap DataParallel so state dict can be loaded on any device
        raw_model = model.module if hasattr(model, "module") else model
        ckpt = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if extra:
            ckpt.update(extra)

        path = self.checkpoint_dir / f"{self.model_name}_epoch{epoch}.pt"
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path}")

        # Also save as 'best' if this is the best metric
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if not best_path.exists():
            torch.save(ckpt, best_path)
        else:
            prev = torch.load(best_path, map_location="cpu", weights_only=False)
            prev_f1 = prev.get("metrics", {}).get("val_f1_weighted", 0)
            curr_f1 = metrics.get("val_f1_weighted", 0)
            if curr_f1 > prev_f1:
                torch.save(ckpt, best_path)
                logger.info(f"New best model saved (F1: {prev_f1:.4f} -> {curr_f1:.4f})")

        self._cleanup()
        return str(path)

    def load(self, model: nn.Module, optimizer=None, scheduler=None, path: Optional[str] = None):
        """Load a checkpoint with automatic DataParallel prefix alignment."""
        if path is None:
            path = str(self.checkpoint_dir / f"{self.model_name}_best.pt")

        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}")
            return None

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        # Adapt module. prefix depending on whether current model is DataParallel
        is_dp = hasattr(model, "module")
        aligned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module.") and not is_dp:
                aligned_state_dict[k[7:]] = v
            elif not k.startswith("module.") and is_dp:
                aligned_state_dict[f"module.{k}"] = v
            else:
                aligned_state_dict[k] = v

        model.load_state_dict(aligned_state_dict)
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
        return ckpt

    def _cleanup(self):
        """Remove old checkpoints, keeping only max_keep most recent."""
        ckpts = sorted(
            self.checkpoint_dir.glob(f"{self.model_name}_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(ckpts) > self.max_keep:
            oldest = ckpts.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")


class EarlyStopping:
    """Early stopping to terminate training when val loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict:
    """Compute accuracy, per-class F1, weighted F1, and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )
    import numpy as np

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    acc = accuracy_score(labels_np, preds_np)
    f1_weighted = f1_score(labels_np, preds_np, average="weighted", zero_division=0)
    f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)
    f1_per_class = f1_score(labels_np, preds_np, average=None, zero_division=0)
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(num_classes)))

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def get_class_weights(label_counts: Dict[str, int], label_map: Dict[str, int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced dataset."""
    import numpy as np
    total = sum(label_counts.values())
    n_classes = len(label_map)
    weights = torch.zeros(n_classes)
    for label, idx in label_map.items():
        count = label_counts.get(label, 1)
        weights[idx] = total / (n_classes * count)
    return weights


def is_kaggle() -> bool:
    """Detect if running in Kaggle environment."""
    return os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_paths(config: Dict) -> Dict[str, str]:
    """Return the correct paths based on environment (local vs Kaggle)."""
    if is_kaggle():
        return config.get("paths", {}).get("kaggle", {})
    return config.get("paths", {}).get("local", {})
