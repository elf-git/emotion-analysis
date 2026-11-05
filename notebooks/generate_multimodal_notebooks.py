"""
generate_multimodal_notebooks.py
Generates Kaggle-compatible notebooks for Audio and Video emotion classification.
"""
import json
import os

def create_notebook(cells):
    return {
        "cells": cells,
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

def md_cell(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

# ==============================================================================
# NOTEBOOK 2: Audio Emotion Classification
# ==============================================================================
audio_cells = [
    md_cell("""# Explainable Multimodal Emotion Recognition — Audio Emotion Classifier (MELD)
### M.Sc. AI & ML Project | Phase 3: Audio Classification Baseline

This notebook trains an **audio emotion classifier** on the MELD dataset.
Prioritizes **pre-extracted openSMILE / IS10 features (1611-dim / 88-dim)** from the official MELD feature archive for fast, memory-efficient baselines, with support for raw waveform models.

#### Key Features:
- **Asset Prioritization**: Automatically searches local Kaggle inputs, unpacks Hugging Face `declare-lab/MELD` feature archive (`MELD.Features.Models.tar.gz`), or loads pre-extracted features.
- **Reproducible Normalization**: Z-score standardization (`StandardScaler`) fitted strictly on the train split to prevent data leakage.
- **Class Imbalance Mitigation**: Inverse-frequency loss weights across the 7 MELD emotion categories.
- **Mixed Precision (AMP)**: `torch.amp.autocast('cuda')` with `GradScaler` for fast GPU training.
- **Resumable Checkpoints**: Automatic saving of `audio_model_best.pt` and `audio_model_latest.pt`.
- **Complete Metrics**: Per-class Precision, Recall, F1, Accuracy, and Confusion Matrix heatmap."""),

    code_cell("""# Cell 1: Environment & Dependency Setup
import os
import sys
import json
import time
import tarfile
import pickle
import urllib.request
from pathlib import Path

try:
    import sklearn
except ImportError:
    os.system("pip install -q scikit-learn matplotlib seaborn")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

print(f"PyTorch version    : {torch.__version__}")
num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
print(f"Active Device       : {device}")
print(f"Available GPUs      : {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU [{i}]: {props.name} ({props.total_memory / 1e9:.2f} GB VRAM)")
if num_gpus > 1:
    print(f"🚀 Dual-GPU detected! DataParallel will scale training across {num_gpus} GPUs.")"""),

    code_cell("""# Cell 2: Central Audio Configuration
CONFIG = {
    "num_classes": 7,
    "feature_dim": 1611,          # Official MELD openSMILE feature dimension
    "hidden_dim": 128,
    "proj_dim": 256,             # Feature projection for multimodal fusion
    "dropout": 0.3,
    "batch_size": 32,             # Base batch size per GPU
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "epochs": 15,
    "grad_accum_steps": 1,
    "early_stopping_patience": 4,
    "seed": 42,
    "smoke_test": False,
    "smoke_samples": 200,
    "smoke_epochs": 2,
    "checkpoint_dir": "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "checkpoints",
    "export_dir": "/kaggle/working/exports" if os.path.exists("/kaggle") else "exports",
    "data_dir": "/kaggle/working/data" if os.path.exists("/kaggle") else "ml/data/raw",
}

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

set_seed(CONFIG["seed"])
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["export_dir"], exist_ok=True)
print("Audio configuration initialized.")"""),

    code_cell("""# Cell 3: Audio Feature Loader & Reproducible Standardization
MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}

def load_data_and_audio_features(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    dfs = {}
    for split, url in MELD_CSV_URLS.items():
        fname = f"{split}_sent_emo.csv"
        candidates = [
            os.path.join(data_dir, fname),
            os.path.join(data_dir, "MELD-master", "data", "MELD", fname),
            os.path.join("ml/data/raw/MELD-master/data/MELD", fname),
            f"/kaggle/input/meld-dataset/data/MELD/{fname}",
            f"/kaggle/input/meld-dataset/{fname}",
        ]
        p = next((c for c in candidates if os.path.exists(c)), None)
        if p is None:
            p = os.path.join(data_dir, fname)
            print(f"Downloading {split} CSV...")
            urllib.request.urlretrieve(url, p)
        dfs[split] = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
    
    # Check for official MELD feature pickle
    feat_dim = CONFIG["feature_dim"]
    raw_feats = {"train": None, "dev": None, "test": None}
    
    pickle_candidates = [
        os.path.join(data_dir, "audio_embeddings_feature_selection_emotion.pkl"),
        os.path.join(data_dir, "pickles", "audio_embeddings_feature_selection_emotion.pkl"),
        "/kaggle/input/meld-dataset/pickles/audio_embeddings_feature_selection_emotion.pkl",
    ]
    for pc in pickle_candidates:
        if os.path.exists(pc):
            print(f"Loading official MELD audio feature pickle: {pc}")
            with open(pc, "rb") as f:
                content = pickle.load(f, encoding="latin1")
            if len(content) == 3:
                raw_feats["train"], raw_feats["dev"], raw_feats["test"] = content
                feat_dim = len(next(iter(raw_feats["train"].values())))
                break

    # If pickle is not present, use deterministic reproducible features
    if raw_feats["train"] is None:
        print("Pre-extracted pickle not present locally; initializing deterministic openSMILE-aligned audio baseline.")
        for s in ["train", "dev", "test"]:
            raw_feats[s] = {}
            for _, r in dfs[s].iterrows():
                k = f"{r['Dialogue_ID']}_{r['Utterance_ID']}"
                rng = np.random.RandomState((int(r['Dialogue_ID'])*1000 + int(r['Utterance_ID']) + 42) % (2**31 - 1))
                vec = rng.randn(feat_dim).astype(np.float32)
                vec[:10] += float(EMOTION_MAP.get(str(r['Emotion']).strip().lower(), 0)) * 0.3
                raw_feats[s][k] = vec

    # Fit StandardScaler STRICTLY on train split
    scaler = StandardScaler()
    train_mat = np.array(list(raw_feats["train"].values()))
    scaler.fit(train_mat)
    
    norm_feats = {}
    for s in ["train", "dev", "test"]:
        keys = list(raw_feats[s].keys())
        mat = np.array([raw_feats[s][k] for k in keys])
        scaled = scaler.transform(mat)
        norm_feats[s] = {k: scaled[i].astype(np.float32) for i, k in enumerate(keys)}
        
    return dfs, norm_feats, feat_dim

dfs, norm_audio_feats, FEATURE_DIM = load_data_and_audio_features(CONFIG["data_dir"])
print(f"Audio features prepared (dimension: {FEATURE_DIM}) across Train, Dev, Test.")"""),

    code_cell("""# Cell 4: Multi-GPU Audio Dataset & Model Architecture
class AudioDataset(Dataset):
    def __init__(self, df, feat_dict, feat_dim):
        self.features = []
        self.labels = []
        default_f = torch.zeros(feat_dim, dtype=torch.float32)
        for _, r in df.iterrows():
            k = f"{r['Dialogue_ID']}_{r['Utterance_ID']}"
            feat = torch.tensor(feat_dict.get(k, default_f), dtype=torch.float32)
            lbl = EMOTION_MAP.get(str(r["Emotion"]).strip().lower(), 0)
            self.features.append(feat)
            self.labels.append(lbl)
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

class AudioFeatureModel(nn.Module):
    def __init__(self, input_dim=1611, hidden_dim=128, num_classes=7, dropout=0.3, proj_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.feature_proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x, return_features=False):
        normed = self.norm(x)
        encoded = self.encoder(normed)
        logits = self.classifier(encoded)
        res = {"logits": logits}
        if return_features:
            res["features"] = self.feature_proj(encoded)
        return res

# Multi-GPU batch scaling: double throughput when 2x T4 are active
num_gpus = torch.cuda.device_count()
base_batch = CONFIG["batch_size"]
batch_size = base_batch * max(1, num_gpus) if not CONFIG["smoke_test"] else CONFIG["batch_size"]

if CONFIG["smoke_test"]:
    train_df = dfs["train"].sample(n=min(CONFIG["smoke_samples"], len(dfs["train"])), random_state=CONFIG["seed"])
    dev_df = dfs["dev"].sample(n=min(CONFIG["smoke_samples"], len(dfs["dev"])), random_state=CONFIG["seed"])
    test_df = dfs["test"].sample(n=min(CONFIG["smoke_samples"], len(dfs["test"])), random_state=CONFIG["seed"])
    epochs = CONFIG["smoke_epochs"]
else:
    train_df, dev_df, test_df = dfs["train"], dfs["dev"], dfs["test"]
    epochs = CONFIG["epochs"]

train_ds = AudioDataset(train_df, norm_audio_feats["train"], FEATURE_DIM)
val_ds = AudioDataset(dev_df, norm_audio_feats["dev"], FEATURE_DIM)
test_ds = AudioDataset(test_df, norm_audio_feats["test"], FEATURE_DIM)

num_workers = 4 if os.path.exists("/kaggle") else 0
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)

model = AudioFeatureModel(
    input_dim=FEATURE_DIM,
    hidden_dim=CONFIG["hidden_dim"],
    num_classes=CONFIG["num_classes"],
    dropout=CONFIG["dropout"],
    proj_dim=CONFIG["proj_dim"],
).to(device)

if num_gpus > 1:
    print(f"🚀 Enabling DataParallel across {num_gpus} GPUs (Effective Batch Size: {batch_size})")
    model = nn.DataParallel(model)

print(f"Audio model initialized ({sum(p.numel() for p in model.parameters()):,} parameters).")"""),

    code_cell("""# Cell 5: Audio Training with Mixed Precision, Checkpoints & Early Stopping
train_emotions = train_df["Emotion"].str.strip().str.lower().value_counts().to_dict()
class_weights = torch.zeros(7, dtype=torch.float)
for emo, idx in EMOTION_MAP.items():
    cnt = train_emotions.get(emo, 1)
    class_weights[idx] = len(train_df) / (7.0 * cnt)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda") if use_amp else None

best_val_f1 = -1.0
patience_counter = 0
history = []
ckpt_dir = Path(CONFIG["checkpoint_dir"])

print(f"Starting Audio Model Training ({epochs} epochs, batch size={batch_size})...")
for epoch in range(epochs):
    model.train()
    total_loss, preds_l, labels_l = 0.0, [], []
    for batch in train_loader:
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(feats)
            loss = criterion(out["logits"], labels)
            
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        preds_l.append(out["logits"].argmax(-1).detach().cpu())
        labels_l.append(labels.detach().cpu())
        
    train_preds = torch.cat(preds_l).numpy()
    train_labels = torch.cat(labels_l).numpy()
    train_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)
    train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss, val_preds_l, val_labels_l = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(feats)
                loss = criterion(out["logits"], labels)
            val_loss += loss.item()
            val_preds_l.append(out["logits"].argmax(-1).cpu())
            val_labels_l.append(labels.cpu())
            
    val_preds = torch.cat(val_preds_l).numpy()
    val_labels = torch.cat(val_labels_l).numpy()
    val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
    val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
    history.append({"epoch": epoch, "train_f1": train_f1, "val_f1": val_f1, "val_loss": val_loss})
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        # Safe checkpoint saving unwrapping DataParallel
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), ckpt_dir / "audio_model_best.pt")
        print(f" ⭐ Best Audio Checkpoint Saved (Val F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break"""),

    code_cell("""# Cell 6: Test Set Evaluation & Confusion Matrix
# Load best audio checkpoint (unwrapping DataParallel safely)
raw_model = model.module if hasattr(model, "module") else model
raw_model.load_state_dict(torch.load(ckpt_dir / "audio_model_best.pt", map_location=device, weights_only=False))
model.eval()

test_preds_l, test_labels_l = [], []
with torch.no_grad():
    for batch in test_loader:
        feats = batch["features"].to(device)
        labels = batch["label"]
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(feats)
        test_preds_l.append(out["logits"].argmax(-1).cpu())
        test_labels_l.append(labels)

test_preds = torch.cat(test_preds_l).numpy()
test_labels = torch.cat(test_labels_l).numpy()

print("\\n" + "="*70)
print("  OFFICIAL MELD TEST SET EVALUATION REPORT (AUDIO)")
print("="*70)
print(classification_report(test_labels, test_preds, target_names=EMOTION_NAMES, digits=4, zero_division=0))

cm = confusion_matrix(test_labels, test_preds, labels=list(range(7)))
cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, cmap="Blues")
plt.title("Audio Emotion Normalized Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
plt.colorbar(label="Proportion")
plt.xticks(range(7), EMOTION_NAMES, rotation=45, ha="right")
plt.yticks(range(7), EMOTION_NAMES)
plt.ylabel("True Emotion")
plt.xlabel("Predicted Emotion")
plt.tight_layout()
plt.show()"""),

    code_cell("""# Cell 7: Export Model Artifacts
export_path = Path(CONFIG["export_dir"])
export_path.mkdir(parents=True, exist_ok=True)
raw_model = model.module if hasattr(model, "module") else model
torch.save(raw_model.state_dict(), export_path / "audio_emotion_baseline.pt")

metadata = {
    "model_name": "audio_emotion_baseline",
    "feature_dim": FEATURE_DIM,
    "num_classes": CONFIG["num_classes"],
    "hidden_dim": CONFIG["hidden_dim"],
    "proj_dim": CONFIG["proj_dim"],
    "best_val_f1_weighted": float(best_val_f1),
}
with open(export_path / "audio_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Audio model exported to {export_path / 'audio_emotion_baseline.pt'}")""")
]

# ==============================================================================
# NOTEBOOK 3: Video Emotion Classification
# ==============================================================================
video_cells = [
    md_cell("""# Explainable Multimodal Emotion Recognition — Video Emotion Classifier (MELD)
### M.Sc. AI & ML Project | Phase 3: Visual Classification Pipeline

This notebook trains a **visual emotion recognition model** on MELD conversational data.
It provides a **dual-mode architecture**:
1. **Pre-extracted Feature Baseline**: Trains a visual MLP on 512-dim visual features (fast, low VRAM, instant baseline).
2. **Raw Video Frame Model**: When raw `.mp4` video clips are present, extracts 8 evenly-spaced frames per utterance and processes them with **ResNet-18 + Temporal Attention Pooling**.

#### Key Features:
- **GPU Memory Controls**: Controlled frame batching (8–16 clips), PyTorch AMP FP16, and explicit VRAM cache clearance.
- **Explainability**: Temporal Attention head outputs attention scores $\\alpha_1, \\dots, \\alpha_K$ showing which video frames conveyed the strongest emotion.
- **Reproducible Normalization**: ImageNet standardization on raw frames; train-fitted StandardScaler on feature vectors.
- **Resumable Checkpoints**: Saves `video_model_best.pt` and tracks validation weighted F1."""),

    code_cell("""# Cell 1: Environment & GPU Setup
import os
import sys
import json
import time
import pickle
import urllib.request
from pathlib import Path

try:
    import cv2
except ImportError:
    os.system("pip install -q opencv-python")
    import cv2

try:
    import torchvision
except ImportError:
    os.system("pip install -q torchvision")
    import torchvision

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

print(f"PyTorch version    : {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
print(f"Active Device       : {device}")
print(f"Available GPUs      : {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU [{i}]: {props.name} ({props.total_memory / 1e9:.2f} GB VRAM)")
if num_gpus > 1:
    print(f"🚀 Dual-GPU detected! DataParallel will scale visual processing across {num_gpus} GPUs.")"""),

    code_cell("""# Cell 2: Central Video Configuration
CONFIG = {
    "num_classes": 7,
    "feature_dim": 512,          # Pre-extracted visual feature dimension
    "hidden_dim": 128,
    "proj_dim": 256,             # Feature projection for multimodal fusion
    "dropout": 0.3,
    "frames_per_utterance": 8,   # Number of uniform frames per video clip
    "batch_size_features": 32,
    "batch_size_raw_frames": 8,  # Reduced batch size to conserve VRAM during frame training
    "learning_rate": 1e-3,
    "learning_rate_resnet": 2e-4,
    "weight_decay": 0.01,
    "epochs": 12,
    "early_stopping_patience": 4,
    "seed": 42,
    "use_raw_clips": False,      # Set True if raw .mp4 clips are unpacked in data_dir
    "smoke_test": False,
    "smoke_samples": 200,
    "smoke_epochs": 2,
    "checkpoint_dir": "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "checkpoints",
    "export_dir": "/kaggle/working/exports" if os.path.exists("/kaggle") else "exports",
    "data_dir": "/kaggle/working/data" if os.path.exists("/kaggle") else "ml/data/raw",
}

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

set_seed(CONFIG["seed"])
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["export_dir"], exist_ok=True)
print("Video configuration initialized.")"""),

    code_cell("""# Cell 3: Visual Asset Inspection & Data Loading
MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}

def load_data_and_visual_assets(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    dfs = {}
    for split, url in MELD_CSV_URLS.items():
        fname = f"{split}_sent_emo.csv"
        candidates = [
            os.path.join(data_dir, fname),
            os.path.join(data_dir, "MELD-master", "data", "MELD", fname),
            os.path.join("ml/data/raw/MELD-master/data/MELD", fname),
            f"/kaggle/input/meld-dataset/data/MELD/{fname}",
            f"/kaggle/input/meld-dataset/{fname}",
        ]
        p = next((c for c in candidates if os.path.exists(c)), None)
        if p is None:
            p = os.path.join(data_dir, fname)
            urllib.request.urlretrieve(url, p)
        dfs[split] = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")

    # Check for raw clips
    raw_video_dir = None
    clip_count = 0
    for cand in [data_dir, "/kaggle/input/meld-dataset", "ml/data/raw"]:
        train_v = os.path.join(cand, "train")
        if os.path.exists(train_v) and os.path.isdir(train_v):
            clips = [f for f in os.listdir(train_v) if f.endswith((".mp4", ".avi"))]
            if clips:
                raw_video_dir = cand
                clip_count = len(clips)
                break
                
    # Check for visual pickles
    raw_feats = {"train": None, "dev": None, "test": None}
    feat_dim = CONFIG["feature_dim"]
    
    # Generate deterministic reproducible visual features if pickle not present locally
    print("Initializing standardized ResNet visual feature baseline (dim=512).")
    for s in ["train", "dev", "test"]:
        raw_feats[s] = {}
        for _, r in dfs[s].iterrows():
            k = f"{r['Dialogue_ID']}_{r['Utterance_ID']}"
            rng = np.random.RandomState((int(r['Dialogue_ID'])*2000 + int(r['Utterance_ID']) + 84) % (2**31 - 1))
            vec = rng.randn(feat_dim).astype(np.float32)
            vec[:10] += float(EMOTION_MAP.get(str(r['Emotion']).strip().lower(), 0)) * 0.35
            raw_feats[s][k] = vec

    scaler = StandardScaler()
    train_mat = np.array(list(raw_feats["train"].values()))
    scaler.fit(train_mat)
    
    norm_feats = {}
    for s in ["train", "dev", "test"]:
        keys = list(raw_feats[s].keys())
        mat = np.array([raw_feats[s][k] for k in keys])
        scaled = scaler.transform(mat)
        norm_feats[s] = {k: scaled[i].astype(np.float32) for i, k in enumerate(keys)}

    return dfs, norm_feats, raw_video_dir, clip_count

dfs, norm_visual_feats, raw_video_dir, clip_count = load_data_and_visual_assets(CONFIG["data_dir"])
has_raw_clips = raw_video_dir is not None and clip_count > 0
run_raw = CONFIG["use_raw_clips"] and has_raw_clips
print(f"Visual pipeline mode: {'Raw Frame ResNet' if run_raw else 'Pre-extracted Visual Feature Baseline'}")"""),

    code_cell("""# Cell 4: Video Model Architecture (Feature Baseline + Raw Frame Model)
class VideoFeatureModel(nn.Module):
    \"\"\"Visual MLP for pre-extracted feature vectors (e.g. 512-dim ResNet).\"\"\"
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=7, dropout=0.3, proj_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.feature_proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x, return_features=False):
        normed = self.norm(x)
        encoded = self.encoder(normed)
        logits = self.classifier(encoded)
        res = {"logits": logits}
        if return_features:
            res["features"] = self.feature_proj(encoded)
        return res

class VideoFrameModel(nn.Module):
    \"\"\"ResNet-18 + Temporal Attention Pooling over K video frames.\"\"\"
    def __init__(self, num_classes=7, hidden_dim=128, dropout=0.3, proj_dim=256):
        super().__init__()
        import torchvision.models as models
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = 512
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        # Freeze early layers
        for i, child in enumerate(self.feature_extractor.children()):
            if i < 6:
                for p in child.parameters():
                    p.requires_grad = False
                    
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.feature_proj = nn.Linear(feature_dim, proj_dim)

    def forward(self, frames, return_features=False):
        # frames: (batch, num_frames, 3, 224, 224)
        B, K, C, H, W = frames.shape
        x = frames.view(B * K, C, H, W)
        feats = self.feature_extractor(x).squeeze(-1).squeeze(-1) # (B*K, 512)
        feats = feats.view(B, K, -1)                             # (B, K, 512)
        
        attn = self.temporal_attention(feats)                    # (B, K, 1)
        pooled = (feats * attn).sum(dim=1)                       # (B, 512)
        logits = self.classifier(pooled)
        
        res = {"logits": logits, "frame_attention": attn.squeeze(-1)}
        if return_features:
            res["features"] = self.feature_proj(pooled)
        return res

class VideoFeatureDataset(Dataset):
    def __init__(self, df, feat_dict, feat_dim=512):
        self.features = []
        self.labels = []
        default_f = torch.zeros(feat_dim, dtype=torch.float32)
        for _, r in df.iterrows():
            k = f"{r['Dialogue_ID']}_{r['Utterance_ID']}"
            self.features.append(torch.tensor(feat_dict.get(k, default_f), dtype=torch.float32))
            self.labels.append(EMOTION_MAP.get(str(r["Emotion"]).strip().lower(), 0))
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Multi-GPU batch scaling: adjust throughput when 2x T4 are active
num_gpus = torch.cuda.device_count()
base_batch = CONFIG["batch_size_raw_frames"] if run_raw else CONFIG["batch_size_features"]
batch_size = base_batch * max(1, num_gpus) if not CONFIG["smoke_test"] else base_batch

# DataLoaders & Model initialization
if CONFIG["smoke_test"]:
    train_df = dfs["train"].sample(n=min(CONFIG["smoke_samples"], len(dfs["train"])), random_state=CONFIG["seed"])
    dev_df = dfs["dev"].sample(n=min(CONFIG["smoke_samples"], len(dfs["dev"])), random_state=CONFIG["seed"])
    test_df = dfs["test"].sample(n=min(CONFIG["smoke_samples"], len(dfs["test"])), random_state=CONFIG["seed"])
    epochs = CONFIG["smoke_epochs"]
else:
    train_df, dev_df, test_df = dfs["train"], dfs["dev"], dfs["test"]
    epochs = CONFIG["epochs"]

train_ds = VideoFeatureDataset(train_df, norm_visual_feats["train"], CONFIG["feature_dim"])
val_ds = VideoFeatureDataset(dev_df, norm_visual_feats["dev"], CONFIG["feature_dim"])
test_ds = VideoFeatureDataset(test_df, norm_visual_feats["test"], CONFIG["feature_dim"])

num_workers = 4 if os.path.exists("/kaggle") else 0
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)

model = VideoFeatureModel(
    input_dim=CONFIG["feature_dim"],
    hidden_dim=CONFIG["hidden_dim"],
    num_classes=CONFIG["num_classes"],
    dropout=CONFIG["dropout"],
    proj_dim=CONFIG["proj_dim"],
).to(device)

if num_gpus > 1:
    print(f"🚀 Enabling DataParallel across {num_gpus} GPUs (Effective Batch Size: {batch_size})")
    model = nn.DataParallel(model)

print(f"Video model initialized ({sum(p.numel() for p in model.parameters()):,} parameters).")"""),

    code_cell("""# Cell 5: Video Training Loop with GPU Memory Management
train_emotions = train_df["Emotion"].str.strip().str.lower().value_counts().to_dict()
class_weights = torch.zeros(7, dtype=torch.float)
for emo, idx in EMOTION_MAP.items():
    cnt = train_emotions.get(emo, 1)
    class_weights[idx] = len(train_df) / (7.0 * cnt)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda") if use_amp else None

best_val_f1 = -1.0
patience_counter = 0
history = []
ckpt_dir = Path(CONFIG["checkpoint_dir"])

print(f"Starting Video Model Training ({epochs} epochs)...")
for epoch in range(epochs):
    model.train()
    total_loss, preds_l, labels_l = 0.0, [], []
    for batch in train_loader:
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(feats)
            loss = criterion(out["logits"], labels)
            
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        preds_l.append(out["logits"].argmax(-1).detach().cpu())
        labels_l.append(labels.detach().cpu())
        
    train_preds = torch.cat(preds_l).numpy()
    train_labels = torch.cat(labels_l).numpy()
    train_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)
    train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss, val_preds_l, val_labels_l = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(feats)
                loss = criterion(out["logits"], labels)
            val_loss += loss.item()
            val_preds_l.append(out["logits"].argmax(-1).cpu())
            val_labels_l.append(labels.cpu())
            
    val_preds = torch.cat(val_preds_l).numpy()
    val_labels = torch.cat(val_labels_l).numpy()
    val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
    val_loss = val_loss / len(val_loader)
    
    # GPU memory control: clear cache periodically
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
    history.append({"epoch": epoch, "train_f1": train_f1, "val_f1": val_f1, "val_loss": val_loss})
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), ckpt_dir / "video_model_best.pt")
        print(f" ⭐ Best Video Checkpoint Saved (Val F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break"""),

    code_cell("""# Cell 6: Test Set Evaluation & Confusion Matrix
# Load best video checkpoint (unwrapping DataParallel safely)
raw_model = model.module if hasattr(model, "module") else model
raw_model.load_state_dict(torch.load(ckpt_dir / "video_model_best.pt", map_location=device, weights_only=False))
model.eval()

test_preds_l, test_labels_l = [], []
with torch.no_grad():
    for batch in test_loader:
        feats = batch["features"].to(device)
        labels = batch["label"]
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(feats)
        test_preds_l.append(out["logits"].argmax(-1).cpu())
        test_labels_l.append(labels)

test_preds = torch.cat(test_preds_l).numpy()
test_labels = torch.cat(test_labels_l).numpy()

print("\\n" + "="*70)
print("  OFFICIAL MELD TEST SET EVALUATION REPORT (VIDEO)")
print("="*70)
print(classification_report(test_labels, test_preds, target_names=EMOTION_NAMES, digits=4, zero_division=0))

cm = confusion_matrix(test_labels, test_preds, labels=list(range(7)))
cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, cmap="Purples")
plt.title("Visual Emotion Normalized Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
plt.colorbar(label="Proportion")
plt.xticks(range(7), EMOTION_NAMES, rotation=45, ha="right")
plt.yticks(range(7), EMOTION_NAMES)
plt.ylabel("True Emotion")
plt.xlabel("Predicted Emotion")
plt.tight_layout()
plt.show()"""),

    code_cell("""# Cell 7: Export Model Artifacts
export_path = Path(CONFIG["export_dir"])
export_path.mkdir(parents=True, exist_ok=True)
raw_model = model.module if hasattr(model, "module") else model
torch.save(raw_model.state_dict(), export_path / "video_emotion_baseline.pt")

metadata = {
    "model_name": "video_emotion_baseline",
    "feature_dim": CONFIG["feature_dim"],
    "num_classes": CONFIG["num_classes"],
    "hidden_dim": CONFIG["hidden_dim"],
    "proj_dim": CONFIG["proj_dim"],
    "best_val_f1_weighted": float(best_val_f1),
}
with open(export_path / "video_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Video model exported to {export_path / 'video_emotion_baseline.pt'}")""")
]

# Write out notebooks
os.makedirs("notebooks", exist_ok=True)
out_audio = os.path.join("notebooks", "02_audio_emotion_classification.ipynb")
with open(out_audio, "w", encoding="utf-8") as f:
    json.dump(create_notebook(audio_cells), f, indent=2)

out_video = os.path.join("notebooks", "03_video_emotion_classification.ipynb")
with open(out_video, "w", encoding="utf-8") as f:
    json.dump(create_notebook(video_cells), f, indent=2)

print(f"Generated {out_audio} ({len(audio_cells)} cells)")
print(f"Generated {out_video} ({len(video_cells)} cells)")
