"""
__init__.py - Models package
Exposes all model classes for convenient import.
"""
from ml.models.text_model import TextEmotionModel
from ml.models.audio_model import AudioFeatureModel, AudioWav2VecModel
from ml.models.video_model import VideoFrameModel, VideoFeatureModel
from ml.models.fusion_model import MultimodalFusionModel, BimodalFusionModel

__all__ = [
    "TextEmotionModel",
    "AudioFeatureModel",
    "AudioWav2VecModel",
    "VideoFrameModel",
    "VideoFeatureModel",
    "MultimodalFusionModel",
    "BimodalFusionModel",
]
