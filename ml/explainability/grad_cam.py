"""
grad_cam.py - Visual Grad-CAM & Feature Attribution for Emotion Recognition
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to highlight salient visual regions in video frames driving emotion decisions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List


class GradCAM:
    """
    Grad-CAM for ResNet / CNN backbones.
    Hooks into target convolutional layer (e.g. layer4) to produce 2D saliency maps.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Find target layer if not provided (default to last conv block in ResNet)
        if target_layer is None:
            target_layer = self._find_target_layer(model)

        self.target_layer = target_layer
        if target_layer is not None:
            self.hooks.append(target_layer.register_forward_hook(self._forward_hook))
            self.hooks.append(target_layer.register_full_backward_hook(self._backward_hook))

    def _find_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Automatically search for the last convolutional layer in the backbone."""
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        frames: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for input video frames.

        Args:
            frames: (1, K, 3, 224, 224) or (1, 3, 224, 224) frame tensor
            target_class: Optional target emotion index

        Returns:
            cam_maps: (K, 224, 224) or (224, 224) normalized heatmaps in [0, 1]
            predicted_class: Predicted emotion index
            confidence: Probability score for predicted class
        """
        device = frames.device
        self.model.zero_grad()

        # Forward pass
        out = self.model(frames)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        confidence = float(probs[0, target_class].item())

        # Backward pass on target score
        target_score = logits[0, target_class]
        target_score.backward()

        if self.activations is None or self.gradients is None:
            # Fallback if layer hooks didn't trigger
            shape = frames.shape[-2:]
            return np.ones(shape, dtype=np.float32) * 0.5, target_class, confidence

        # Global average pool the gradients over spatial dimensions (H, W)
        # activations: (B, C, H, W), gradients: (B, C, H, W)
        weights = self.gradients.mean(dim=[-2, -1], keepdim=True) # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True) # (B, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input frame resolution
        target_size = (frames.shape[-2], frames.shape[-1])
        cam_upsampled = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

        cam_np = cam_upsampled.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max - c_min > 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, confidence

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class VisualFeatureAttributor:
    """
    Feature-level gradient saliency for pre-extracted visual MLP baselines.
    Identifies which of the 512 visual features contributed most strongly to the emotion.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def attribute(self, features: torch.Tensor, target_class: Optional[int] = None) -> Dict:
        """
        Compute input-gradient saliency: S_i = |x_i * (dY / dx_i)|
        """
        feat_var = features.clone().detach().requires_grad_(True)
        out = self.model(feat_var)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        saliency = (feat_var * feat_var.grad).abs().squeeze(0).detach().cpu().numpy()
        saliency_norm = saliency / (saliency.sum() + 1e-9)

        return {
            "predicted_class": target_class,
            "confidence": confidence,
            "feature_saliency": saliency_norm,
            "top_features": np.argsort(saliency_norm)[::-1][:10].tolist(),
        }


def overlay_cam_on_frame(frame_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap onto a video frame (BGR format for OpenCV / matplotlib).
    """
    try:
        import cv2
        cam_scaled = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_scaled, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay
    except ImportError:
        # Fallback if cv2 not installed
        cam_3ch = np.stack([cam, 1 - cam, np.zeros_like(cam)], axis=-1)
        blended = (1 - alpha) * (frame_bgr.astype(np.float32) / 255.0) + alpha * cam_3ch
        return np.uint8(np.clip(blended * 255.0, 0, 255))
