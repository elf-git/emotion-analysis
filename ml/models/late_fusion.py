"""
late_fusion.py - Late Fusion Architecture & Ensembling
Combines decision-level predictions from unimodal text, audio, and visual branches.
Supports weighted average, confidence pooling, and learned meta-classifier stacking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class LateFusionModel(nn.Module):
    """
    Late Fusion Model combining predictions from unimodal branches.
    Supports arbitrary subsets of modalities (e.g. T, A, V, T+A, T+V, A+V, T+A+V).
    """

    def __init__(
        self,
        num_classes: int = 7,
        modalities: Optional[List[str]] = None,
        fusion_strategy: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            num_classes: Number of emotion classes (7 for MELD)
            modalities: List of modality names ['text', 'audio', 'video']
            fusion_strategy: 'weighted_average', 'max_confidence', or 'stacking'
            weights: Optional manual weights dict per modality {'text': 0.35, 'audio': 0.30, 'video': 0.35}
        """
        super().__init__()
        self.num_classes = num_classes
        self.modalities = modalities or ["text", "audio", "video"]
        self.fusion_strategy = fusion_strategy

        # Learnable or fixed weights
        if weights is not None:
            w_list = [weights.get(m, 1.0 / len(self.modalities)) for m in self.modalities]
            total = sum(w_list)
            self.weights = torch.tensor([w / total for w in w_list], dtype=torch.float32)
        else:
            self.weights = torch.ones(len(self.modalities), dtype=torch.float32) / len(self.modalities)

        # Meta-classifier for stacking (optional)
        if fusion_strategy == "stacking":
            self.meta_classifier = nn.Sequential(
                nn.Linear(num_classes * len(self.modalities), 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

    def forward(self, branch_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            branch_logits: Dict mapping modality name -> logits tensor (batch, num_classes)

        Returns:
            Dict containing:
                'logits': (batch, num_classes) fused logits
                'probs': (batch, num_classes) fused probabilities
                'weights': (len(modalities),) effective modality weights
        """
        active_mods = [m for m in self.modalities if m in branch_logits and branch_logits[m] is not None]
        if not active_mods:
            raise ValueError(f"None of requested modalities {self.modalities} provided in input.")

        device = next(iter(branch_logits.values())).device
        batch_size = next(iter(branch_logits.values())).shape[0]

        # Calculate softmax probabilities per modality
        branch_probs = {m: F.softmax(branch_logits[m], dim=-1) for m in active_mods}

        if self.fusion_strategy == "weighted_average":
            weights = self.weights.to(device)
            # Re-normalize weights for active modalities
            active_weights = torch.tensor(
                [self.weights[self.modalities.index(m)] for m in active_mods],
                device=device,
                dtype=torch.float32
            )
            active_weights = active_weights / (active_weights.sum() + 1e-9)

            fused_probs = torch.zeros((batch_size, self.num_classes), device=device)
            for m, w in zip(active_mods, active_weights):
                fused_probs += w * branch_probs[m]

            # Compute log probabilities as logits
            fused_logits = torch.log(fused_probs + 1e-9)

            return {
                "logits": fused_logits,
                "probs": fused_probs,
                "weights": active_weights,
            }

        elif self.fusion_strategy == "max_confidence":
            # Pick prediction from modality that exhibits highest max-class confidence
            all_probs = torch.stack([branch_probs[m] for m in active_mods], dim=1) # (B, M, C)
            max_conf, _ = all_probs.max(dim=-1) # (B, M)
            best_mod_idx = max_conf.argmax(dim=-1) # (B,)
            
            fused_probs = all_probs[torch.arange(batch_size), best_mod_idx]
            fused_logits = torch.log(fused_probs + 1e-9)

            return {
                "logits": fused_logits,
                "probs": fused_probs,
                "selected_modality_idx": best_mod_idx,
            }

        elif self.fusion_strategy == "stacking":
            concat_logits = torch.cat([branch_logits[m] for m in active_mods], dim=-1)
            fused_logits = self.meta_classifier(concat_logits)
            fused_probs = F.softmax(fused_logits, dim=-1)

            return {
                "logits": fused_logits,
                "probs": fused_probs,
            }

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")


def optimize_late_fusion_weights(
    val_branch_probs: Dict[str, np.ndarray],
    val_labels: np.ndarray,
    modalities: List[str],
    grid_steps: int = 21,
) -> Tuple[Dict[str, float], float]:
    """
    Grid-search optimal late fusion weights on validation split to maximize Weighted F1.
    
    Args:
        val_branch_probs: Dict mapping modality -> (N_val, num_classes) numpy probability array
        val_labels: (N_val,) true emotion indices
        modalities: Modalities to include in ablation/fusion
        grid_steps: Granularity of search

    Returns:
        (best_weights_dict, best_val_f1)
    """
    from sklearn.metrics import f1_score

    if len(modalities) == 1:
        m = modalities[0]
        preds = val_branch_probs[m].argmax(axis=-1)
        f1 = f1_score(val_labels, preds, average="weighted", zero_division=0)
        return {m: 1.0}, float(f1)

    elif len(modalities) == 2:
        m1, m2 = modalities
        best_f1 = -1.0
        best_w = (0.5, 0.5)
        for w1 in np.linspace(0.0, 1.0, grid_steps):
            w2 = 1.0 - w1
            fused = w1 * val_branch_probs[m1] + w2 * val_branch_probs[m2]
            preds = fused.argmax(axis=-1)
            score = f1_score(val_labels, preds, average="weighted", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_w = (float(w1), float(w2))
        return {m1: best_w[0], m2: best_w[1]}, float(best_f1)

    elif len(modalities) == 3:
        m1, m2, m3 = modalities
        best_f1 = -1.0
        best_w = (0.33, 0.33, 0.34)
        for w1 in np.linspace(0.05, 0.90, grid_steps):
            for w2 in np.linspace(0.05, 0.95 - w1, grid_steps):
                w3 = 1.0 - (w1 + w2)
                if w3 < 0:
                    continue
                fused = w1 * val_branch_probs[m1] + w2 * val_branch_probs[m2] + w3 * val_branch_probs[m3]
                preds = fused.argmax(axis=-1)
                score = f1_score(val_labels, preds, average="weighted", zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_w = (float(w1), float(w2), float(w3))
        return {m1: best_w[0], m2: best_w[1], m3: best_w[2]}, float(best_f1)

    raise ValueError(f"Unsupported number of modalities: {len(modalities)}")
