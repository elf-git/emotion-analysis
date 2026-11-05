"""
audio_model.py - Audio-based Emotion Recognition Model
Supports two modes:
  1. Pre-extracted features (openSMILE / MELD pickle features)
  2. Raw waveform via Wav2Vec2 (if raw audio is available)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class AudioFeatureModel(nn.Module):
    """
    Audio emotion classifier using pre-extracted features.
    
    Designed for:
        - openSMILE eGeMAPSv02 features (88-dim)
        - MELD provided features (1611-dim for emotion)
        - Any fixed-dimension audio feature vector
    
    Architecture:
        Feature Vector -> BatchNorm -> FC layers -> Emotion Logits
    """

    def __init__(
        self,
        input_dim: int = 1611,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.3,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

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

    def forward(
        self,
        audio_features: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = self.norm(audio_features)
        encoded = self.encoder(x)  # (batch, hidden_dim)
        logits = self.classifier(encoded)

        result = {"logits": logits}
        if return_features:
            result["features"] = self.feature_proj(encoded)

        return result
        """
        Args:
            audio_features: (batch, input_dim) pre-extracted audio features
            return_features: If True, return projected features for fusion

        Returns:
            Dict with 'logits', optionally 'features'
        """
        x = self.bn(audio_features)
        encoded = self.encoder(x)  # (batch, hidden_dim)
        logits = self.classifier(encoded)

        result = {"logits": logits}
        if return_features:
            result["features"] = self.feature_proj(encoded)

        return result


class AudioWav2VecModel(nn.Module):
    """
    Audio emotion classifier using Wav2Vec2 for raw waveform input.
    Use this only when raw .wav files are available.
    
    Architecture:
        Raw Waveform -> Wav2Vec2 -> Mean Pooling -> FC -> Emotion Logits
    """

    def __init__(
        self,
        backbone: str = "facebook/wav2vec2-base",
        num_classes: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained(backbone)
        encoder_dim = self.encoder.config.hidden_size  # 768

        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.feature_proj = nn.Linear(encoder_dim, hidden_dim)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: (batch, waveform_length) raw audio
            attention_mask: (batch, waveform_length), optional
            return_features: If True, return projected features for fusion

        Returns:
            Dict with 'logits', optionally 'features'
        """
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, time, encoder_dim)

        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.classifier(pooled)
        result = {"logits": logits}
        if return_features:
            result["features"] = self.feature_proj(pooled)
        return result
