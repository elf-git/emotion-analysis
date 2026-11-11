"""
pretrained_models.py - Real emotion inference for text, voice and faces.

Runs public pretrained models. These are NOT trained by this project:
  - Text:  j-hartmann/emotion-english-distilroberta-base (7 emotions, English)
  - Face:  trpakov/vit-face-expression (7 emotions, Vision Transformer)
  - Voice: superb/wav2vec2-base-superb-er (4 emotions: neutral, joy, anger, sadness)

A modality with no usable input returns None. It is reported as missing and is
never replaced with a guessed value or treated as neutral.
"""
import base64
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(PROJECT_ROOT / ".model_cache")

EMOTION_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMOTION_MAP = {name: idx for idx, name in enumerate(EMOTION_NAMES)}

TEXT_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
FACE_MODEL_ID = "trpakov/vit-face-expression"
VOICE_MODEL_ID = "superb/wav2vec2-base-superb-er"

MODEL_INFO = {
    "text": {"model": TEXT_MODEL_ID, "architecture": "DistilRoBERTa", "classes": 7},
    "visual": {"model": FACE_MODEL_ID, "architecture": "Vision Transformer (ViT)", "classes": 7},
    "audio": {"model": VOICE_MODEL_ID, "architecture": "wav2vec 2.0", "classes": 4},
}

# Late-fusion weights, renormalised over whichever modalities are present.
# Set by hand, not fitted on labelled data. Voice is low because the voice model
# labels white noise as joy at 99%, so it is unreliable on film audio.
FUSION_WEIGHTS = {"text": 0.40, "audio": 0.15, "visual": 0.45}

LABEL_ALIASES = {
    "neu": "neutral", "neutral": "neutral",
    "hap": "joy", "happy": "joy", "happiness": "joy", "joy": "joy",
    "ang": "anger", "angry": "anger", "anger": "anger",
    "sad": "sadness", "sadness": "sadness",
    "fear": "fear", "surprise": "surprise", "disgust": "disgust",
}

SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 0.4
SILENCE_RMS = 0.003
MAX_VOICE_SEC = 30

_load_lock = threading.Lock()
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _hub_options(repo_id: str) -> Dict[str, Any]:
    """Uses the local copy without going online when a model is already downloaded."""
    from huggingface_hub import try_to_load_from_cache

    cached = isinstance(try_to_load_from_cache(repo_id, "config.json", cache_dir=CACHE_DIR), str)
    return {"cache_dir": CACHE_DIR, "local_files_only": cached}


def warm_up() -> None:
    """Loads every model so the first analysis does not pay the start-up cost."""
    TextEmotionModel._load()
    FaceEmotionModel._load()
    VoiceEmotionModel._load()


def _to_seven_classes(probs: np.ndarray, id2label: Dict[int, str]) -> np.ndarray:
    """Maps a model's own label order onto this project's 7-emotion order."""
    out = np.zeros(len(EMOTION_NAMES), dtype=np.float32)
    for idx, value in enumerate(probs):
        name = LABEL_ALIASES.get(str(id2label[idx]).lower())
        if name is None:
            raise ValueError(f"Unmapped emotion label: {id2label[idx]}")
        out[EMOTION_MAP[name]] += float(value)
    return out


def models_available() -> Dict[str, bool]:
    """Reports which pretrained models are present in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    return {
        name: isinstance(try_to_load_from_cache(info["model"], "config.json", cache_dir=CACHE_DIR), str)
        for name, info in MODEL_INFO.items()
    }


class TextEmotionModel:
    """English text emotion classifier with Integrated Gradients token attribution."""

    _tokenizer = None
    _model = None

    @classmethod
    def _load(cls) -> None:
        if cls._model is not None:
            return
        with _load_lock:
            if cls._model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                cls._tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, **_hub_options(TEXT_MODEL_ID))
                cls._model = AutoModelForSequenceClassification.from_pretrained(
                    TEXT_MODEL_ID, use_safetensors=False, **_hub_options(TEXT_MODEL_ID)
                ).to(_device).eval()
                logger.info(f"Loaded text emotion model {TEXT_MODEL_ID}")

    @classmethod
    def predict(cls, text: str) -> np.ndarray:
        cls._load()
        enc = cls._tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(_device)
        with torch.no_grad():
            probs = torch.softmax(cls._model(**enc).logits[0], dim=-1).cpu().numpy()
        return _to_seven_classes(probs, cls._model.config.id2label)

    @classmethod
    def _target_logit_index(cls, emotion: str) -> int:
        for idx, label in cls._model.config.id2label.items():
            if LABEL_ALIASES.get(str(label).lower()) == emotion:
                return int(idx)
        raise ValueError(f"Model has no class for {emotion}")

    @classmethod
    def explain(cls, text: str, emotion: str, steps: int = 16) -> List[Dict[str, Any]]:
        """Integrated Gradients from a padding baseline, summed per whitespace word."""
        cls._load()
        enc = cls._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, return_offsets_mapping=True
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(_device)
        attention_mask = enc["attention_mask"].to(_device)

        embedding_layer = cls._model.get_input_embeddings()
        with torch.no_grad():
            inputs = embedding_layer(input_ids)
            pad_ids = torch.full_like(input_ids, cls._tokenizer.pad_token_id)
            special = torch.tensor(
                [start == end for start, end in offsets], device=_device
            ).unsqueeze(0)
            baseline = embedding_layer(torch.where(special, input_ids, pad_ids))

        alphas = torch.linspace(1.0 / steps, 1.0, steps, device=_device).view(-1, 1, 1)
        path = (baseline + alphas * (inputs - baseline)).detach().requires_grad_(True)
        logits = cls._model(
            inputs_embeds=path, attention_mask=attention_mask.expand(steps, -1)
        ).logits
        target = cls._target_logit_index(emotion)
        grads = torch.autograd.grad(logits[:, target].sum(), path)[0]
        token_scores = ((inputs - baseline) * grads.mean(dim=0, keepdim=True)).sum(dim=-1)[0]
        token_scores = token_scores.detach().cpu().numpy()

        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
        word_scores = np.zeros(len(words), dtype=np.float64)
        for (start, end), score in zip(offsets, token_scores):
            if start == end:
                continue
            for idx, (w_start, w_end, _) in enumerate(words):
                if start < w_end and end > w_start:
                    word_scores[idx] += float(score)
                    break

        total = float(np.abs(word_scores).sum()) or 1.0
        result = [
            {
                "token": word,
                "score": round(abs(float(value)) / total, 4),
                "importance_score": round(abs(float(value)) / total, 4),
                "importance_pct": round(abs(float(value)) / total * 100.0, 1),
                "direction": "supports" if value >= 0 else "opposes",
            }
            for (_, _, word), value in zip(words, word_scores)
        ]
        result.sort(key=lambda item: item["score"], reverse=True)
        return result


class FaceEmotionModel:
    """Face detection (OpenCV) plus facial expression classification (ViT)."""

    _processor = None
    _model = None
    _detectors: List[Any] = []

    @classmethod
    def _load(cls) -> None:
        if cls._model is not None:
            return
        with _load_lock:
            if cls._model is None:
                from transformers import AutoImageProcessor, AutoModelForImageClassification

                cls._processor = AutoImageProcessor.from_pretrained(FACE_MODEL_ID, **_hub_options(FACE_MODEL_ID))
                cls._model = AutoModelForImageClassification.from_pretrained(
                    FACE_MODEL_ID, **_hub_options(FACE_MODEL_ID)
                ).to(_device).eval()
                base = cv2.data.haarcascades
                cls._detectors = [
                    cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
                    cv2.CascadeClassifier(base + "haarcascade_profileface.xml"),
                ]
                logger.info(f"Loaded face emotion model {FACE_MODEL_ID}")

    @classmethod
    def detect_largest_face(cls, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns a padded crop of the largest detected face, or None."""
        cls._load()
        height, width = frame_bgr.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        min_side = max(40, int(min(height, width) * 0.08))

        boxes: List[Tuple[int, int, int, int]] = []
        for detector in cls._detectors:
            found = detector.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
            boxes.extend(tuple(int(v) for v in box) for box in found)
        # The profile cascade only finds faces turned one way, so also try the mirror image.
        mirrored = cls._detectors[1].detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=(min_side, min_side))
        boxes.extend((width - int(x) - int(w), int(y), int(w), int(h)) for x, y, w, h in mirrored)

        if not boxes:
            return None
        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        pad = int(0.12 * max(w, h))
        return frame_bgr[max(0, y - pad): min(height, y + h + pad), max(0, x - pad): min(width, x + w + pad)]

    @classmethod
    def _inputs(cls, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops_bgr]
        return cls._processor(images=rgb, return_tensors="pt")["pixel_values"].to(_device)

    @classmethod
    def predict_crops(cls, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Returns one 7-class probability row per face crop."""
        cls._load()
        with torch.no_grad():
            probs = torch.softmax(cls._model(pixel_values=cls._inputs(crops_bgr)).logits, dim=-1).cpu().numpy()
        return np.stack([_to_seven_classes(row, cls._model.config.id2label) for row in probs])

    @classmethod
    def gradcam(cls, crop_bgr: np.ndarray, emotion: str) -> Optional[str]:
        """Grad-CAM over the ViT's patch tokens, returned as a base64 PNG overlay."""
        cls._load()
        target = next(
            (int(i) for i, label in cls._model.config.id2label.items()
             if LABEL_ALIASES.get(str(label).lower()) == emotion),
            None,
        )
        if target is None:
            return None

        pixel_values = cls._inputs([crop_bgr]).requires_grad_(True)
        outputs = cls._model(pixel_values=pixel_values, output_hidden_states=True)
        hidden = outputs.hidden_states[-2]
        grads = torch.autograd.grad(outputs.logits[0, target], hidden)[0]

        patches, patch_grads = hidden[0, 1:], grads[0, 1:]
        channel_weights = patch_grads.mean(dim=0)
        cam = torch.relu((patches * channel_weights).sum(dim=-1)).detach().cpu().numpy()
        side = int(round(len(cam) ** 0.5))
        if side * side != len(cam):
            return None
        cam = cam.reshape(side, side)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

        face = cv2.resize(crop_bgr, (224, 224))
        heat = cv2.resize(cam.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face, 0.6, heatmap, 0.4, 0)
        ok, buffer = cv2.imencode(".png", overlay)
        if not ok:
            return None
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


class VoiceEmotionModel:
    """Speech emotion classifier. Only covers neutral, joy, anger and sadness."""

    _extractor = None
    _model = None

    @classmethod
    def _load(cls) -> None:
        if cls._model is not None:
            return
        with _load_lock:
            if cls._model is None:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                cls._extractor = AutoFeatureExtractor.from_pretrained(VOICE_MODEL_ID, **_hub_options(VOICE_MODEL_ID))
                cls._model = AutoModelForAudioClassification.from_pretrained(
                    VOICE_MODEL_ID, use_safetensors=False, **_hub_options(VOICE_MODEL_ID)
                ).to(_device).eval()
                logger.info(f"Loaded voice emotion model {VOICE_MODEL_ID}")

    @classmethod
    def predict(cls, wave_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Returns 7-class probabilities, or None for silence or too little audio."""
        if wave_data is None or len(wave_data) < MIN_AUDIO_SEC * SAMPLE_RATE:
            return None
        if float(np.sqrt(np.mean(wave_data ** 2))) < SILENCE_RMS:
            return None
        cls._load()
        clip = wave_data[: MAX_VOICE_SEC * SAMPLE_RATE]
        inputs = cls._extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(_device)
        with torch.no_grad():
            probs = torch.softmax(cls._model(**inputs).logits[0], dim=-1).cpu().numpy()
        return _to_seven_classes(probs, cls._model.config.id2label)


def load_audio(path: str) -> Optional[np.ndarray]:
    """Decodes any audio or video file to 16 kHz mono float32 with FFmpeg."""
    command = [
        "ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except Exception as exc:
        logger.warning(f"Could not decode audio from {path}: {exc}")
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def compute_prosody(wave_data: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """Measures pitch (autocorrelation), RMS energy and an energy-peak speaking rate."""
    if wave_data is None or len(wave_data) < MIN_AUDIO_SEC * SAMPLE_RATE:
        return None
    from scipy.signal import find_peaks

    frame = int(0.04 * SAMPLE_RATE)
    hop = int(0.02 * SAMPLE_RATE)
    min_lag, max_lag = SAMPLE_RATE // 400, SAMPLE_RATE // 60
    overall_rms = float(np.sqrt(np.mean(wave_data ** 2)))

    pitches, energies = [], []
    for start in range(0, len(wave_data) - frame, hop):
        chunk = wave_data[start: start + frame]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)
        if rms < max(SILENCE_RMS, 0.5 * overall_rms):
            continue
        chunk = chunk - chunk.mean()
        corr = np.correlate(chunk, chunk, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if corr[lag] / corr[0] > 0.5:
            pitches.append(SAMPLE_RATE / lag)

    duration = len(wave_data) / SAMPLE_RATE
    energy_curve = np.array(energies)
    peaks, _ = find_peaks(energy_curve, height=max(SILENCE_RMS, 0.6 * overall_rms), distance=6)
    return {
        "pitch_hz": round(float(np.median(pitches)), 1) if pitches else None,
        "rms_energy": round(overall_rms, 4),
        "speaking_rate_syllables_per_sec": round(len(peaks) / duration, 1),
        "voiced_ratio": round(len(pitches) / max(1, len(energies)), 2),
    }


def fuse(modality_probs: Dict[str, Optional[np.ndarray]]) -> Optional[Dict[str, Any]]:
    """
    Weighted late fusion of probability vectors over the modalities that are present.
    Returns None when no modality has evidence.
    """
    present = {name: probs for name, probs in modality_probs.items() if probs is not None}
    if not present:
        return None

    weight_total = sum(FUSION_WEIGHTS[name] for name in present)
    fused = sum((FUSION_WEIGHTS[name] / weight_total) * probs for name, probs in present.items())
    fused = fused / fused.sum()
    pred_idx = int(np.argmax(fused))

    # A modality's contribution is how much probability it gave the winning emotion.
    support = {name: FUSION_WEIGHTS[name] / weight_total * float(probs[pred_idx]) for name, probs in present.items()}
    support_total = sum(support.values()) or 1.0
    shares = {name: support.get(name, 0.0) / support_total * 100.0 for name in FUSION_WEIGHTS}
    labels = {"text": "Linguistic (Text)", "audio": "Acoustic (Audio)", "visual": "Visual (Video Face)"}

    return {
        "probabilities": fused.astype(np.float32),
        "predicted_index": pred_idx,
        "predicted_emotion": EMOTION_NAMES[pred_idx],
        "confidence": float(fused[pred_idx]),
        "contributions": {
            "text_pct": round(shares["text"], 1),
            "audio_pct": round(shares["audio"], 1),
            "video_pct": round(shares["visual"], 1),
            "dominant_modality": labels[max(support, key=support.get)],
        },
    }
