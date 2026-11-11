"""
inference_service.py - Production Multimodal Inference & Uncertainty Quantification
Runs the pretrained text, voice and face models and fuses their real probabilities,
with uncertainty analysis, modality contributions, and token/visual explainability.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pretrained_models import MODEL_INFO, TextEmotionModel, fuse, models_available

logger = logging.getLogger(__name__)

EMOTION_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMOTION_MAP = {name: idx for idx, name in enumerate(EMOTION_NAMES)}
ID_TO_EMOTION = {idx: name for idx, name in enumerate(EMOTION_NAMES)}

EMOTION_COLORS = {
    "neutral": "#94a3b8",   # Slate Gray
    "surprise": "#38bdf8",  # Sky Blue
    "fear": "#c084fc",      # Purple
    "sadness": "#60a5fa",   # Blue
    "joy": "#fbbf24",       # Amber / Gold
    "disgust": "#34d399",   # Emerald
    "anger": "#f87171",     # Rose / Red
}

EMOTION_VALENCE = {
    "neutral": 0.0,
    "joy": 0.85,
    "surprise": 0.35,
    "sadness": -0.75,
    "fear": -0.65,
    "disgust": -0.70,
    "anger": -0.80,
}


def compute_uncertainty_metrics(probs: np.ndarray) -> Dict[str, Any]:
    """
    Computes Shannon Entropy, Top-2 Confidence Margin, and Confidence Intervals.
    """
    K = len(probs)
    # Normalized Shannon Entropy in [0, 1]
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / np.log(K)
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if K > 1 else 0.0
    margin = float(top1 - top2)

    if entropy < 0.35 and margin > 0.40:
        level = "Low (Decisive Prediction)"
        color = "#10b981"  # Green
    elif entropy < 0.65 and margin > 0.15:
        level = "Moderate (Consistent with Mild Ambiguity)"
        color = "#f59e0b"  # Amber
    else:
        level = "High (Cross-Modal Conflict / High Ambiguity)"
        color = "#ef4444"  # Red

    # 95% Confidence Interval for top prediction (Wald approximation)
    margin_err = 1.96 * np.sqrt(max(0.0, (top1 * (1.0 - top1)) / 100.0))
    ci_lower = max(0.0, top1 - margin_err)
    ci_upper = min(1.0, top1 + margin_err)

    return {
        "entropy": round(entropy, 4),
        "entropy_normalized_pct": round(entropy * 100.0, 1),
        "top2_margin": round(margin, 4),
        "uncertainty_level": level,
        "level_color": color,
        "confidence_interval_95": [round(ci_lower, 4), round(ci_upper, 4)],
    }


class MultimodalInferenceEngine:
    """
    Singleton service combining the pretrained modality models into one prediction.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model_status(self) -> Dict[str, bool]:
        return models_available()

    @property
    def is_loaded(self) -> bool:
        return all(self.model_status.values())

    def predict_single(
        self,
        text: Optional[str] = None,
        audio_features: Optional[np.ndarray] = None,
        video_features: Optional[np.ndarray] = None,
        audio_summary: Optional[Dict] = None,
        video_summary: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Fuses text, voice and face predictions.
        audio_features and video_features are 7-class probability vectors.
        """
        has_text = bool(text and text.strip())
        has_audio = audio_features is not None
        has_video = video_features is not None
        if not (has_text or has_audio or has_video):
            raise ValueError("At least one usable modality (text, audio, or video) must be provided.")

        text_probs = TextEmotionModel.predict(text) if has_text else None
        fusion = fuse({"text": text_probs, "audio": audio_features, "visual": video_features})
        probs = fusion["probabilities"]
        pred_emotion = fusion["predicted_emotion"]
        confidence = fusion["confidence"]

        token_attributions = TextEmotionModel.explain(text, pred_emotion) if has_text else []
        uncertainty = compute_uncertainty_metrics(probs)

        prob_dict = {
            EMOTION_NAMES[i]: {
                "probability": round(float(probs[i]), 4),
                "percentage": round(float(probs[i]) * 100.0, 1),
                "color": EMOTION_COLORS[EMOTION_NAMES[i]],
            }
            for i in range(len(EMOTION_NAMES))
        }

        return {
            "predicted_emotion": pred_emotion,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "emotion_color": EMOTION_COLORS[pred_emotion],
            "valence_score": EMOTION_VALENCE[pred_emotion],
            "probabilities": prob_dict,
            "modality_contributions": fusion["contributions"],
            "uncertainty": uncertainty,
            "explainability": {
                "token_attributions": token_attributions[:6],
                "prosody_summary": audio_summary,
                "visual_summary": video_summary,
            },
            "active_modalities": {
                "text": has_text,
                "audio": has_audio,
                "video": has_video,
            },
            "models": MODEL_INFO,
        }

    def predict_dialogue_timeline(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Runs conversational dialogue emotion tracking over a sequence of speaker utterances.
        Produces an emotional timeline showing emotion shifts and valence drift.
        """
        timeline = []
        emotion_counts = {e: 0 for e in EMOTION_NAMES}

        for idx, turn in enumerate(turns):
            text = turn.get("text", "")
            speaker = turn.get("speaker", f"Speaker {idx % 2 + 1}")
            res = self.predict_single(text=text)

            emotion = res["predicted_emotion"]
            emotion_counts[emotion] += 1

            timeline.append({
                "turn_index": idx + 1,
                "speaker": speaker,
                "text": text,
                "emotion": emotion,
                "emotion_color": res["emotion_color"],
                "confidence_pct": res["confidence_pct"],
                "valence_score": res["valence_score"],
                "dominant_modality": res["modality_contributions"]["dominant_modality"],
                "uncertainty_level": res["uncertainty"]["uncertainty_level"],
            })

        avg_valence = round(float(np.mean([t["valence_score"] for t in timeline])), 2) if timeline else 0.0

        return {
            "total_turns": len(timeline),
            "average_valence": avg_valence,
            "conversational_tone": "Constructive / Positive" if avg_valence > 0.1 else ("Tense / Conflictual" if avg_valence < -0.1 else "Neutral / Balanced"),
            "emotion_distribution": emotion_counts,
            "timeline": timeline,
        }
