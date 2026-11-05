"""
video_model.py - Video/Visual Emotion Recognition Model
Extracts visual features from video frames using a pretrained CNN.
Supports both pre-extracted features and raw frame input.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class VideoFrameModel(nn.Module):
    """
    Video emotion classifier using frame-level CNN features.
    
    Architecture:
        Frames -> ResNet18 (frozen early layers) -> Temporal Attention -> FC -> Logits
    
    For each utterance, we sample K frames and aggregate via attention.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pretrained: bool = True,
        frames_per_utterance: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.frames_per_utterance = frames_per_utterance

        import torchvision.models as models

        if backbone == "resnet18":
            base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # Freeze early layers (first 6 of 8 children)
        for i, child in enumerate(self.feature_extractor.children()):
            if i < 6:
                for param in child.parameters():
                    param.requires_grad = False

        # Temporal attention over frames
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.feature_proj = nn.Linear(feature_dim, 256)

    def forward(
        self,
        frames: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_frames, C, H, W = frames.shape
        x = frames.view(batch_size * num_frames, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.squeeze(-1).squeeze(-1)
        feats = feats.view(batch_size, num_frames, -1)

        attn = self.temporal_attention(feats)
        pooled = (feats * attn).sum(dim=1)
        logits = self.classifier(pooled)

        result = {
            "logits": logits,
            "frame_attention": attn.squeeze(-1),
        }
        if return_features:
            result["features"] = self.feature_proj(pooled)
        return result


class VideoFeatureModel(nn.Module):
    """
    Video emotion classifier using pre-extracted visual features.
    Use this when raw video frames are not available but features are.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.3,
        proj_dim: int = 256,
    ):
        super().__init__()
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
        video_features: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        normed = self.norm(video_features)
        encoded = self.encoder(normed)
        logits = self.classifier(encoded)

        result = {"logits": logits}
        if return_features:
            result["features"] = self.feature_proj(encoded)
        return result
