"""
text_model.py - Text-based Emotion Recognition Model
Uses BERT (or DistilBERT for smoke tests) with attention-based classification.
Supports explainability via attention weight extraction.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class TextEmotionModel(nn.Module):
    """
    Text encoder for emotion recognition.
    
    Architecture:
        BERT/DistilBERT -> Attention Pooling -> FC -> Emotion Logits
    
    Explainability:
        - Extracts attention weights from the last transformer layer
        - Returns token-level importance scores
    """

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_layers: int = 8,
        return_attention: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.return_attention = return_attention
        self.num_classes = num_classes

        # Lazy import to avoid loading transformers at module level
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(backbone, output_attentions=return_attention)
        self.encoder = AutoModel.from_pretrained(backbone, config=config)
        encoder_dim = config.hidden_size  # 768 for bert-base

        # Freeze early layers to save memory
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

        # Attention pooling over sequence
        self.attention_pool = nn.Sequential(
            nn.Linear(encoder_dim, 1),
            nn.Softmax(dim=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Feature projection (for fusion)
        self.feature_proj = nn.Linear(encoder_dim, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len), optional
            return_features: If True, also return projected features for fusion

        Returns:
            Dict with keys:
                'logits': (batch, num_classes)
                'attention_weights': (batch, num_heads, seq_len, seq_len) if return_attention
                'features': (batch, hidden_dim) if return_features
                'token_importance': (batch, seq_len) - aggregated attention scores
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "distilbert" not in self.backbone_name.lower():
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, encoder_dim)

        # Attention pooling
        attn_weights = self.attention_pool(hidden_states)  # (batch, seq_len, 1)
        pooled = (hidden_states * attn_weights).sum(dim=1)  # (batch, encoder_dim)

        logits = self.classifier(pooled)

        result = {
            "logits": logits,
            "token_importance": attn_weights.squeeze(-1),  # (batch, seq_len)
        }

        if self.return_attention and hasattr(outputs, "attentions") and outputs.attentions:
            # Last layer attention: (batch, num_heads, seq_len, seq_len)
            result["attention_weights"] = outputs.attentions[-1]

        if return_features:
            result["features"] = self.feature_proj(pooled)  # (batch, hidden_dim)

        return result

    def get_smoke_test_config(self) -> Dict:
        """Return a lightweight config for smoke testing."""
        return {
            "backbone": "distilbert-base-uncased",
            "num_classes": self.num_classes,
            "hidden_dim": 128,
            "dropout": 0.1,
            "freeze_layers": 4,
        }
