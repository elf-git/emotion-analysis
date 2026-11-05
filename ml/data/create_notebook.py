"""
generate_text_notebook.py
Generates the complete, Kaggle-compatible 01_text_emotion_classification.ipynb notebook.
"""
import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(content):
    lines = [l + "\n" for l in content.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(content):
    lines = [l + "\n" for l in content.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# ==========================================
# Cell 1: Header & Overview
# ==========================================
add_md("""# Explainable Multimodal Emotion Recognition — Text Emotion Classifier (MELD)
### M.Sc. AI & ML Project | Phase 2: Compact Text Classification Baseline

This notebook provides a **complete, self-contained, Kaggle-ready text emotion classification pipeline** trained on the official **MELD** (Multimodal EmotionLines Dataset) conversational dataset.

---

### Pipeline Architecture & Key Features:
1. **Zero-Setup Data Fetching**: Automatically resolves dataset files from Kaggle inputs, local disk, or directly downloads verified official MELD CSVs from GitHub/HuggingFace via URL.
2. **Official Splits & No Leakage**: Preserves official Train (9,989), Dev (1,109), and Test (2,610) splits.
3. **Compact Pretrained Backbone**: Uses `distilbert-base-uncased` (66M parameters) with frozen lower layers for fast convergence and low memory footprint (~40% smaller and 60% faster than BERT-base).
4. **Class Imbalance Mitigation**: Computes inverse-frequency class weights for the 7 emotion classes (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`).
5. **Configurable Mini-Batches & Gradient Accumulation**: Supports mini-batch tuning with gradient accumulation for effective batch size 64 without CUDA Out-Of-Memory.
6. **Mixed Precision (AMP)**: Native PyTorch `torch.amp.autocast('cuda')` with `GradScaler` for fast GPU acceleration.
7. **Checkpoint Saving & Resumption**: Periodically saves latest and best model checkpoints with full optimizer, scheduler, scaler, and epoch state.
8. **Early Stopping**: Prevents overfitting with configurable patience monitoring validation weighted F1.
9. **Explainability**: Custom Attention Pooling head calculates token-level attribution scores to highlight which words drove the emotion prediction.
10. **Comprehensive Metrics & Export**: Computes Accuracy, Weighted F1, Macro F1, Per-class F1, Confusion Matrix heatmap, and exports model artifacts for deployment.""")

# ==========================================
# Cell 2: Environment Detection & Dependencies
# ==========================================
add_code("""# ============================================================
# Cell 1: Environment Detection & Dependencies
# ============================================================
import os
import sys
import json
import time
import urllib.request
from pathlib import Path

# Auto-install transformers and scikit-learn if missing (for Kaggle/Colab)
try:
    import transformers
except ImportError:
    print("Installing transformers...")
    os.system("pip install -q transformers>=4.30.0")
    import transformers

try:
    import sklearn
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install -q scikit-learn")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"PyTorch version    : {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device       : {device}")
if torch.cuda.is_available():
    print(f"GPU Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running in CPU mode (Mixed precision AMP disabled).")""")

# ==========================================
# Cell 3: Configuration & Hyperparameters
# ==========================================
add_code("""# ============================================================
# Cell 2: Central Configuration & Hyperparameters
# ============================================================
CONFIG = {
    # Pretrained Backbone
    "backbone": "distilbert-base-uncased",
    "num_classes": 7,
    "max_seq_length": 128,
    "hidden_dim": 256,
    "dropout": 0.3,
    "freeze_layers": 2,          # Freeze early transformer layers (set 0 to fine-tune all)
    
    # Training Hyperparameters
    "batch_size": 32,             # Mini-batch size (adjust if VRAM is constrained)
    "grad_accum_steps": 2,        # Gradient accumulation (effective batch size = 64)
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "epochs": 8,                 # Full training epochs
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "seed": 42,
    
    # Smoke Test Mode (set True to verify pipeline on a small subset in <1 minute)
    "smoke_test": False,
    "smoke_samples": 200,
    "smoke_epochs": 2,
    "smoke_batch_size": 8,
    
    # Resumption
    "resume_checkpoint": None,    # Set to path (e.g. "checkpoints/text_model_best.pt") to resume
    
    # Environment & Paths
    "checkpoint_dir": "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "checkpoints",
    "export_dir": "/kaggle/working/exports" if os.path.exists("/kaggle") else "exports",
    "data_dir": "/kaggle/working/data" if os.path.exists("/kaggle") else "ml/data/raw",
}

# Emotion Mappings (Official MELD 7-class taxonomy)
EMOTION_MAP = {
    "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
    "joy": 4, "disgust": 5, "anger": 6,
}
ID_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}
EMOTION_NAMES = [ID_TO_EMOTION[i] for i in range(7)]

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(CONFIG["seed"])
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["export_dir"], exist_ok=True)
print(f"Config initialized. Checkpoint directory: {CONFIG['checkpoint_dir']}")""")

# ==========================================
# Cell 4: URL Dataset Fetching & Verification
# ==========================================
add_code("""# ============================================================
# Cell 3: URL Dataset Fetching & Verification
# ============================================================
# Official MELD repository raw URLs
MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}

def resolve_or_download_meld(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    
    candidate_dirs = [
        data_dir,
        os.path.join(data_dir, "MELD-master", "data", "MELD"),
        os.path.join(data_dir, "raw", "MELD-master", "data", "MELD"),
        "ml/data/raw/MELD-master/data/MELD",
        "/kaggle/input/meld-dataset/data/MELD",
        "/kaggle/input/meld-dataset",
        "/kaggle/input/meld",
    ]
    
    for split, url in MELD_CSV_URLS.items():
        fname = f"{split}_sent_emo.csv"
        found = None
        for cdir in candidate_dirs:
            p = os.path.join(cdir, fname)
            if os.path.exists(p):
                found = p
                break
                
        if found is None:
            dest = os.path.join(data_dir, fname)
            print(f"Downloading {split} CSV from URL: {url} -> {dest}...")
            urllib.request.urlretrieve(url, dest)
            found = dest
        else:
            print(f"Using local {split} CSV: {found}")
            
        paths[split] = found
    return paths

csv_paths = resolve_or_download_meld(CONFIG["data_dir"])

# Load into DataFrames with encoding error handling
dfs = {split: pd.read_csv(p, encoding="utf-8", encoding_errors="replace") for split, p in csv_paths.items()}

print("\\n" + "="*60)
print("  MELD DATASET VERIFICATION")
print("="*60)
for split, df in dfs.items():
    print(f"  {split.upper():5s} split: {len(df):5d} utterances | {df['Dialogue_ID'].nunique():4d} dialogues")
    assert "Utterance" in df.columns, f"Missing 'Utterance' in {split}"
    assert "Emotion" in df.columns, f"Missing 'Emotion' in {split}"

# Verify expected official counts
assert len(dfs["train"]) == 9989, f"Expected 9,989 train samples, got {len(dfs['train'])}"
assert len(dfs["dev"]) == 1109, f"Expected 1,109 dev samples, got {len(dfs['dev'])}"
assert len(dfs["test"]) == 2610, f"Expected 2,610 test samples, got {len(dfs['test'])}"
print("\\n✅ All 13,708 utterances across official Train/Dev/Test splits verified without discrepancy.")""")

# ==========================================
# Cell 5: Class Imbalance & PyTorch Dataset
# ==========================================
add_code("""# ============================================================
# Cell 4: Class Weights & PyTorch Dataset Pipeline
# ============================================================
from transformers import AutoTokenizer

# Compute Inverse-Frequency Class Weights
train_emotions = dfs["train"]["Emotion"].str.strip().str.lower().value_counts().to_dict()
total_train = len(dfs["train"])
class_weights = torch.zeros(len(EMOTION_MAP), dtype=torch.float)

print("--- Training Emotion Distribution & Class Weights ---")
for emo, idx in EMOTION_MAP.items():
    cnt = train_emotions.get(emo, 1)
    weight = total_train / (len(EMOTION_MAP) * cnt)
    class_weights[idx] = weight
    print(f"  {emo:10s} (ID {idx}): {cnt:5d} utterances ({cnt/total_train*100:5.2f}%) | Weight: {weight:.2f}")

class MELDTextDataset(Dataset):
    \"\"\"PyTorch Dataset for MELD Text Emotion Classification.\"\"\"
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["Utterance"].astype(str).tolist()
        self.emotions = [EMOTION_MAP.get(str(e).strip().lower(), 0) for e in df["Emotion"]]
        self.dialogue_ids = df["Dialogue_ID"].tolist()
        self.utterance_ids = df["Utterance_ID"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
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
            "label": torch.tensor(self.emotions[idx], dtype=torch.long),
            "text": text,
            "dialogue_id": self.dialogue_ids[idx],
            "utterance_id": self.utterance_ids[idx],
        }

tokenizer = AutoTokenizer.from_pretrained(CONFIG["backbone"])

# Prepare subsets if smoke test is active
if CONFIG["smoke_test"]:
    print(f"\\n*** SMOKE TEST MODE: Truncating to {CONFIG['smoke_samples']} samples per split ***")
    train_df = dfs["train"].sample(n=min(CONFIG["smoke_samples"], len(dfs["train"])), random_state=CONFIG["seed"])
    dev_df = dfs["dev"].sample(n=min(CONFIG["smoke_samples"], len(dfs["dev"])), random_state=CONFIG["seed"])
    test_df = dfs["test"].sample(n=min(CONFIG["smoke_samples"], len(dfs["test"])), random_state=CONFIG["seed"])
    batch_size = CONFIG["smoke_batch_size"]
    epochs = CONFIG["smoke_epochs"]
else:
    train_df, dev_df, test_df = dfs["train"], dfs["dev"], dfs["test"]
    batch_size = CONFIG["batch_size"]
    epochs = CONFIG["epochs"]

train_ds = MELDTextDataset(train_df, tokenizer, CONFIG["max_seq_length"])
val_ds = MELDTextDataset(dev_df, tokenizer, CONFIG["max_seq_length"])
test_ds = MELDTextDataset(test_df, tokenizer, CONFIG["max_seq_length"])

num_workers = 2 if os.path.exists("/kaggle") else 0
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)

print(f"\\nDataLoaders: Train={len(train_loader)} batches | Val={len(val_loader)} batches | Test={len(test_loader)} batches")""")

# ==========================================
# Cell 6: Model Architecture & Attention Pooling
# ==========================================
add_code("""# ============================================================
# Cell 5: Compact Pretrained Model with Explainable Attention
# ============================================================
from transformers import AutoModel, AutoConfig

class TextEmotionModel(nn.Module):
    \"\"\"
    Text Emotion Classifier:
    Pretrained Transformer (DistilBERT) -> Attention Pooling -> MLP Classifier
    Outputs emotion logits and token-level attribution weights for explainability.
    \"\"\"
    def __init__(self, backbone="distilbert-base-uncased", num_classes=7, hidden_dim=256, dropout=0.3, freeze_layers=2):
        super().__init__()
        self.backbone_name = backbone
        config = AutoConfig.from_pretrained(backbone, output_attentions=True)
        self.encoder = AutoModel.from_pretrained(backbone, config=config)
        encoder_dim = config.hidden_size  # 768 for DistilBERT
        
        # Freeze early transformer layers to preserve generic language features and accelerate training
        if freeze_layers > 0:
            modules = list(self.encoder.named_parameters())
            for name, param in modules:
                layer_num = None
                if "layer." in name:
                    try:
                        layer_num = int(name.split("layer.")[1].split(".")[0])
                    except (IndexError, ValueError):
                        pass
                if layer_num is not None and layer_num < freeze_layers:
                    param.requires_grad = False
                elif "embeddings" in name:
                    param.requires_grad = False
                    
        # Explainable Attention Pooling Head
        self.attention_pool = nn.Sequential(
            nn.Linear(encoder_dim, 1),
            nn.Softmax(dim=1),
        )
        
        # Emotion Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Latent projection for future multimodal fusion
        self.feature_proj = nn.Linear(encoder_dim, hidden_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_features=False):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "distilbert" not in self.backbone_name.lower():
            kwargs["token_type_ids"] = token_type_ids
            
        outputs = self.encoder(**kwargs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Masked attention pooling over non-padding tokens
        attn_raw = self.attention_pool(hidden_states)  # (batch, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        attn_weights = attn_raw * mask_expanded
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        pooled = (hidden_states * attn_weights).sum(dim=1)  # (batch, 768)
        logits = self.classifier(pooled)
        
        result = {
            "logits": logits,
            "token_importance": attn_weights.squeeze(-1),
        }
        if hasattr(outputs, "attentions") and outputs.attentions:
            result["attention_weights"] = outputs.attentions[-1]
        if return_features:
            result["features"] = self.feature_proj(pooled)
        return result

model = TextEmotionModel(
    backbone=CONFIG["backbone"],
    num_classes=CONFIG["num_classes"],
    hidden_dim=CONFIG["hidden_dim"],
    dropout=CONFIG["dropout"],
    freeze_layers=CONFIG["freeze_layers"],
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Architecture : {CONFIG['backbone']}")
print(f"Total Parameters   : {total_params:,}")
print(f"Trainable Params   : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")""")

# ==========================================
# Cell 7: Checkpoint Manager & Resumption Setup
# ==========================================
add_code("""# ============================================================
# Cell 6: Optimizer, Checkpoint Manager & Resumption Setup
# ============================================================
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import get_cosine_schedule_with_warmup

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

total_steps = len(train_loader) * epochs
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

class CheckpointManager:
    def __init__(self, checkpoint_dir, model_name="text_model"):
        self.dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_f1 = -1.0
        
    def save(self, model, optimizer, scheduler, scaler, epoch, val_metrics, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "val_metrics": val_metrics,
            "config": CONFIG,
        }
        latest_path = self.dir / f"{self.model_name}_latest.pt"
        torch.save(ckpt, latest_path)
        
        if is_best:
            best_path = self.dir / f"{self.model_name}_best.pt"
            torch.save(ckpt, best_path)
            print(f" ⭐ New Best Checkpoint Saved! (Val Weighted F1: {val_metrics['f1_weighted']:.4f})")
            
    def load(self, model, optimizer=None, scheduler=None, scaler=None, path=None):
        if path is None:
            path = self.dir / f"{self.model_name}_best.pt"
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f" Loaded checkpoint from {path} (Epoch {ckpt.get('epoch', '?')})")
        return ckpt

ckpt_manager = CheckpointManager(CONFIG["checkpoint_dir"], "text_model")
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda") if use_amp else None

start_epoch = 0
history = []
if CONFIG["resume_checkpoint"]:
    ckpt = ckpt_manager.load(model, optimizer, scheduler, scaler, CONFIG["resume_checkpoint"])
    if ckpt:
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch+1}")
else:
    print("Ready to start training from scratch.")""")

# ==========================================
# Cell 8: Training & Validation Loop
# ==========================================
add_code("""# ============================================================
# Cell 7: Training & Validation Loop with Early Stopping
# ============================================================
def compute_metrics(all_preds, all_labels):
    preds_np = all_preds.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "f1_weighted": float(f1_score(labels_np, preds_np, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "f1_per_class": f1_score(labels_np, preds_np, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels_np, preds_np, labels=list(range(7))).tolist(),
    }

def train_epoch(model, loader, optimizer, criterion, scheduler, scaler, grad_accum_steps):
    model.train()
    total_loss = 0.0
    preds_list, labels_list = [], []
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs["logits"], labels) / grad_accum_steps
            
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
                
        total_loss += loss.item() * grad_accum_steps
        preds = outputs["logits"].argmax(dim=-1).detach()
        preds_list.append(preds)
        labels_list.append(labels.detach())
        
    metrics = compute_metrics(torch.cat(preds_list), torch.cat(labels_list))
    metrics["loss"] = total_loss / len(loader)
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds_list, labels_list = [], []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs["logits"], labels)
            
        total_loss += loss.item()
        preds = outputs["logits"].argmax(dim=-1)
        preds_list.append(preds)
        labels_list.append(labels)
        
    metrics = compute_metrics(torch.cat(preds_list), torch.cat(labels_list))
    metrics["loss"] = total_loss / len(loader)
    return metrics

# Execute Training
best_val_f1 = -1.0
patience_counter = 0

print(f"\\n{'='*70}")
print(f"  STARTING TRAINING: {epochs} Epochs | Batch: {batch_size} (Effective: {batch_size*CONFIG['grad_accum_steps']}) | AMP: {use_amp}")
print(f"{'='*70}\\n")

for epoch in range(start_epoch, epochs):
    start_t = time.time()
    train_m = train_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, CONFIG["grad_accum_steps"])
    val_m = evaluate(model, val_loader, criterion)
    duration = time.time() - start_t
    
    is_best = val_m["f1_weighted"] > best_val_f1
    if is_best:
        best_val_f1 = val_m["f1_weighted"]
        patience_counter = 0
    else:
        patience_counter += 1
        
    ckpt_manager.save(model, optimizer, scheduler, scaler, epoch, val_m, is_best=is_best)
    
    record = {"epoch": epoch, "train": train_m, "val": val_m, "time_sec": duration}
    history.append(record)
    
    print(f"Epoch [{epoch+1:02d}/{epochs:02d}] ({duration:.1f}s) | "
          f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} F1-W: {train_m['f1_weighted']:.4f} | "
          f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} F1-W: {val_m['f1_weighted']:.4f}")
          
    if patience_counter >= CONFIG["early_stopping_patience"]:
        print(f"\\n✋ Early stopping triggered! Validation F1 did not improve for {CONFIG['early_stopping_patience']} epochs.")
        break""")

# ==========================================
# Cell 9: Training Curves Visualization
# ==========================================
add_code("""# ============================================================
# Cell 8: Training & Validation Curves Visualization
# ============================================================
if history:
    epochs_range = [h["epoch"] + 1 for h in history]
    train_losses = [h["train"]["loss"] for h in history]
    val_losses = [h["val"]["loss"] for h in history]
    train_f1s = [h["train"]["f1_weighted"] for h in history]
    val_f1s = [h["val"]["f1_weighted"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs_range, train_losses, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, val_losses, "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # F1 plot
    axes[1].plot(epochs_range, train_f1s, "o-", label="Train Weighted F1", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs_range, val_f1s, "s--", label="Val Weighted F1", color="#d62728", linewidth=2)
    axes[1].set_title("Weighted F1 Score vs. Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted F1")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()""")

# ==========================================
# Cell 10: Official Test Evaluation & Confusion Matrix
# ==========================================
add_code("""# ============================================================
# Cell 9: Official Test Split Evaluation & Confusion Matrix
# ============================================================
# Load the best performing checkpoint
ckpt_manager.load(model)
model.eval()

test_preds, test_labels, test_texts = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            
        preds = outputs["logits"].argmax(dim=-1).cpu()
        test_preds.extend(preds.tolist())
        test_labels.extend(labels.tolist())
        test_texts.extend(batch["text"])

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# Print Detailed Classification Report
print("\\n" + "="*70)
print("  OFFICIAL MELD TEST SET EVALUATION REPORT")
print("="*70)
print(classification_report(
    test_labels, test_preds,
    target_names=EMOTION_NAMES,
    digits=4,
    zero_division=0,
))

# Plot Normalized Confusion Matrix Heatmap
cm = confusion_matrix(test_labels, test_preds, labels=list(range(7)))
cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

plt.figure(figsize=(9, 7))
plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
plt.title("Normalized Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
plt.colorbar(label="Proportion")
tick_marks = np.arange(len(EMOTION_NAMES))
plt.xticks(tick_marks, EMOTION_NAMES, rotation=45, ha="right", fontsize=11)
plt.yticks(tick_marks, EMOTION_NAMES, fontsize=11)

thresh = cm_norm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, f"{cm[i, j]}\\n({cm_norm[i, j]*100:.1f}%)",
            horizontalalignment="center", verticalalignment="center",
            color="white" if cm_norm[i, j] > thresh else "black",
            fontsize=9
        )

plt.ylabel("True Emotion", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Emotion", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()""")

# ==========================================
# Cell 11: Explainability Demo
# ==========================================
add_code("""# ============================================================
# Cell 10: Explainability: Token-Level Attention Attribution
# ============================================================
def explain_utterance(text, model, tokenizer, device):
    \"\"\"Extract token attention weights to explain emotion prediction.\"\"\"
    model.eval()
    encoding = tokenizer(
        text,
        max_length=CONFIG["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids, attention_mask)
            
    logits = out["logits"].squeeze(0)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_emotion = ID_TO_EMOTION[pred_idx]
    
    # Token importance from attention pooling
    importance = out["token_importance"].squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    
    # Filter padding / special tokens
    valid_tokens, valid_scores = [], []
    for tok, score in zip(tokens, importance):
        if tok in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
            continue
        valid_tokens.append(tok)
        valid_scores.append(score)
        
    print(f"\\nUtterance : \\\"{text}\\\"")
    print(f"Prediction: {pred_emotion.upper()} ({probs[pred_idx]*100:.1f}%)")
    print("Top Word Attributions:")
    # Sort by importance
    ranked = sorted(zip(valid_tokens, valid_scores), key=lambda x: x[1], reverse=True)[:5]
    for tok, score in ranked:
        bar = "█" * int(score * 40)
        print(f"  {tok:15s} | {score:6.4f} {bar}")

# Test on representative sample utterances
sample_utterances = [
    "Oh my God, that is the most incredible news ever!",
    "I can't believe you did that behind my back, I am so furious with you!",
    "I don't think I can handle this pain anymore, I just want to be alone.",
    "Wait, are you telling me she was here the whole time?!",
    "Yeah, I think the meeting is at four o'clock today.",
]

print("--- Qualitative Token-Level Explainability Demonstrations ---")
for s in sample_utterances:
    explain_utterance(s, model, tokenizer, device)""")

# ==========================================
# Cell 12: Export Artifacts for Web App & Deployment
# ==========================================
add_code("""# ============================================================
# Cell 11: Export Model Artifacts for Web App / Deployment
# ============================================================
export_path = Path(CONFIG["export_dir"])
export_path.mkdir(parents=True, exist_ok=True)

# Save lightweight inference state dict
model_weights_path = export_path / "text_emotion_distilbert.pt"
torch.save(model.state_dict(), model_weights_path)

# Save metadata for frontend/backend loading
metadata = {
    "model_name": "text_emotion_distilbert",
    "backbone": CONFIG["backbone"],
    "num_classes": CONFIG["num_classes"],
    "emotion_map": EMOTION_MAP,
    "id_to_emotion": ID_TO_EMOTION,
    "max_seq_length": CONFIG["max_seq_length"],
    "hidden_dim": CONFIG["hidden_dim"],
    "weights_file": str(model_weights_path.name),
    "best_val_f1_weighted": float(best_val_f1) if best_val_f1 > 0 else None,
}

metadata_path = export_path / "model_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Export Complete!")
print(f"  Model Weights : {model_weights_path} ({os.path.getsize(model_weights_path)/1e6:.1f} MB)")
print(f"  Metadata Spec : {metadata_path}")
print(f"\\nThis export can now be directly loaded by the backend inference API.")""")

# Write output file
out_path = os.path.join("notebooks", "01_text_emotion_classification.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook successfully generated at: {out_path} ({len(notebook['cells'])} cells)")
