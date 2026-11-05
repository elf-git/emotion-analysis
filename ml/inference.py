"""
inference.py - Inference Pipeline
Loads trained models and runs emotion prediction with explainability.
Designed for both batch inference and single-utterance prediction.
"""
import os
import json
import logging
import torch
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
SENTIMENT_LABELS = ["neutral", "positive", "negative"]


class EmotionPredictor:
    """
    High-level inference class for emotion prediction.
    Supports text-only, audio-only, video-only, and multimodal fusion.
    Returns predictions with explainability information.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str = "configs/config.json",
        device: str = "auto",
    ):
        with open(config_path) as f:
            self.config = json.load(f)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_dir = checkpoint_dir
        self.models = {}
        self.tokenizer = None

        self._load_available_models()

    def _load_available_models(self):
        """Load all available trained models."""
        # Text model
        text_ckpt = os.path.join(self.checkpoint_dir, "text_model_best.pt")
        if os.path.exists(text_ckpt):
            from ml.models.text_model import TextEmotionModel
            from transformers import AutoTokenizer

            model = TextEmotionModel(
                backbone=self.config["model"]["text"]["backbone"],
                num_classes=self.config["model"]["fusion"]["num_classes_emotion"],
                hidden_dim=self.config["model"]["text"]["hidden_dim"],
                dropout=0.0,  # No dropout during inference
            )
            ckpt = torch.load(text_ckpt, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            self.models["text"] = model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["text"]["backbone"]
            )
            logger.info("Loaded text model")

        # Fusion model
        fusion_ckpt = os.path.join(self.checkpoint_dir, "fusion_model_best.pt")
        if os.path.exists(fusion_ckpt):
            from ml.models.fusion_model import MultimodalFusionModel
            model = MultimodalFusionModel(
                feature_dim=self.config["model"]["fusion"]["hidden_dim"],
                hidden_dim=self.config["model"]["fusion"]["hidden_dim"],
                num_classes=self.config["model"]["fusion"]["num_classes_emotion"],
                num_heads=self.config["model"]["fusion"]["num_heads"],
                dropout=0.0,
            )
            ckpt = torch.load(fusion_ckpt, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            self.models["fusion"] = model
            logger.info("Loaded fusion model")

    @torch.no_grad()
    def predict_text(self, text: str) -> Dict:
        """
        Predict emotion from text with explainability.

        Returns:
            Dict with 'emotion', 'confidence', 'probabilities',
            'token_importance', and 'attention_weights'
        """
        if "text" not in self.models:
            raise RuntimeError("Text model not loaded. Check checkpoint directory.")

        encoding = self.tokenizer(
            text,
            max_length=self.config["model"]["text"]["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.models["text"](
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

        probs = torch.softmax(outputs["logits"], dim=-1)[0]
        pred_idx = probs.argmax().item()

        # Token importance for explainability
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        token_importance = outputs["token_importance"][0].cpu().tolist()

        # Pair tokens with importance, filter special tokens
        token_scores = [
            {"token": tok, "importance": imp}
            for tok, imp in zip(tokens, token_importance)
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]
        ]

        return {
            "text": text,
            "emotion": EMOTION_LABELS[pred_idx],
            "emotion_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))
            },
            "token_importance": token_scores,
        }

    @torch.no_grad()
    def predict_multimodal(
        self,
        text_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Predict emotion using multimodal fusion with modality importance."""
        if "fusion" not in self.models:
            raise RuntimeError("Fusion model not loaded.")

        modality_features = {}
        if text_features is not None:
            modality_features["text"] = text_features.to(self.device)
        if audio_features is not None:
            modality_features["audio"] = audio_features.to(self.device)
        if video_features is not None:
            modality_features["video"] = video_features.to(self.device)

        outputs = self.models["fusion"](modality_features)
        probs = torch.softmax(outputs["logits"], dim=-1)[0]
        pred_idx = probs.argmax().item()

        modality_weights = outputs["modality_weights"][0].cpu().tolist()

        return {
            "emotion": EMOTION_LABELS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))
            },
            "modality_importance": {
                mod: float(w)
                for mod, w in zip(self.models["fusion"].modalities, modality_weights)
            },
        }

    def get_available_models(self) -> List[str]:
        """Return list of loaded model names."""
        return list(self.models.keys())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--text", type=str, help="Text to predict emotion for")
    args = parser.parse_args()

    predictor = EmotionPredictor(args.checkpoint_dir, args.config)
    print(f"Available models: {predictor.get_available_models()}")

    if args.text:
        result = predictor.predict_text(args.text)
        print(json.dumps(result, indent=2))
