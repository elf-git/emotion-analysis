"""
fusion_model.py - Multimodal Fusion for Emotion Recognition
Combines text, audio, and video features using attention-based fusion.
Supports explainability through modality-level attention weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class CrossModalAttention(nn.Module):
    """Cross-modal attention for pairwise modality interaction."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, 1, dim) - the querying modality
            key_value: (batch, 1, dim) - the attended modality

        Returns:
            (batch, 1, dim) - attended output
        """
        attn_out, _ = self.mha(query, key_value, key_value)
        return self.norm(query + attn_out)


class ModalityGate(nn.Module):
    """
    Learnable gate that produces modality importance weights.
    Provides explainability: which modality contributed most to the prediction.
    """

    def __init__(self, dim: int = 256, num_modalities: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_modalities),
            nn.Softmax(dim=-1),
        )
        self.num_modalities = num_modalities

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of (batch, dim) tensors, one per modality

        Returns:
            (batch, num_modalities) gating weights
        """
        concat = torch.cat(features, dim=-1)  # (batch, dim * num_modalities)
        return self.gate(concat)


class MultimodalFusionModel(nn.Module):
    """
    Attention-based multimodal fusion for emotion recognition.
    
    Architecture:
        Text Features ─┐
        Audio Features ─┼─> Modality Gate -> Weighted Sum -> FC -> Logits
        Video Features ─┘
                         └─> Cross-Modal Attention (optional)
    
    Explainability:
        - Modality gate weights show which modality drives prediction
        - Cross-modal attention shows inter-modality dependencies
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_cross_attention: bool = True,
        modalities: List[str] = None,
    ):
        super().__init__()
        self.modalities = modalities or ["text", "audio", "video"]
        self.num_modalities = len(self.modalities)
        self.use_cross_attention = use_cross_attention
        self.feature_dim = feature_dim

        # Projection layers to align modality dimensions
        self.projections = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
            )
            for mod in self.modalities
        })

        # Cross-modal attention (optional)
        if use_cross_attention:
            self.cross_attns = nn.ModuleDict({
                f"{m1}_{m2}": CrossModalAttention(feature_dim, num_heads, dropout)
                for m1 in self.modalities
                for m2 in self.modalities
                if m1 != m2
            })

        # Modality gate
        self.gate = ModalityGate(feature_dim, self.num_modalities)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_features: Dict mapping modality name to feature tensor (batch, feature_dim).
                               Not all modalities need to be present - missing ones are zero-filled.

        Returns:
            Dict with:
                'logits': (batch, num_classes)
                'modality_weights': (batch, num_modalities) - explainability
                'fused_features': (batch, feature_dim) - for downstream use
        """
        batch_size = None
        device = None

        # Project each modality
        projected = {}
        for mod in self.modalities:
            if mod in modality_features and modality_features[mod] is not None:
                feat = modality_features[mod]
                batch_size = feat.shape[0]
                device = feat.device
                projected[mod] = self.projections[mod](feat)
            else:
                projected[mod] = None

        # Fill missing modalities with zeros
        if batch_size is None:
            raise ValueError("At least one modality must have features")

        for mod in self.modalities:
            if projected[mod] is None:
                projected[mod] = torch.zeros(
                    batch_size, self.feature_dim, device=device
                )

        # Cross-modal attention
        if self.use_cross_attention:
            enhanced = {}
            for mod in self.modalities:
                feat = projected[mod].unsqueeze(1)  # (batch, 1, dim)
                for other_mod in self.modalities:
                    if other_mod != mod:
                        other_feat = projected[other_mod].unsqueeze(1)
                        key = f"{mod}_{other_mod}"
                        if key in self.cross_attns:
                            feat = self.cross_attns[key](feat, other_feat)
                enhanced[mod] = feat.squeeze(1)  # (batch, dim)
        else:
            enhanced = projected

        # Modality gating
        feat_list = [enhanced[mod] for mod in self.modalities]
        gate_weights = self.gate(feat_list)  # (batch, num_modalities)

        # Weighted fusion
        stacked = torch.stack(feat_list, dim=1)  # (batch, num_mod, dim)
        weights = gate_weights.unsqueeze(-1)  # (batch, num_mod, 1)
        fused = (stacked * weights).sum(dim=1)  # (batch, dim)

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "modality_weights": gate_weights,
            "fused_features": fused,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        # Adapt between cross_attentions and cross_attns prefixes seamlessly
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("cross_attentions."):
                new_sd[k.replace("cross_attentions.", "cross_attns.")] = v
            else:
                new_sd[k] = v
        return super().load_state_dict(new_sd, strict=strict)


class BimodalFusionModel(MultimodalFusionModel):
    """Convenience class for text + audio fusion (when video is unavailable)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("modalities", ["text", "audio"])
        super().__init__(**kwargs)
