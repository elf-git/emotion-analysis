"""
ml/explainability - Explainable Multimodal Communication Analysis
Provides Text Attribution (Integrated Gradients & Attention Pooling)
and Visual Grad-CAM for emotion explainability.
"""
from ml.explainability.text_attribution import TextAttributor, explain_text_utterance
from ml.explainability.grad_cam import GradCAM, VisualFeatureAttributor

__all__ = [
    "TextAttributor",
    "explain_text_utterance",
    "GradCAM",
    "VisualFeatureAttributor",
]
