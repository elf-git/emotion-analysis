"""
media_processor.py - Safe Media Validation and Real Voice / Face Emotion Extraction
Enforces file size limits, safe temp file lifecycles, and formats outputs for frontend consumption.
"""
import os
import tempfile
import logging
import cv2
import numpy as np
from typing import Dict

from ml.pretrained_models import (
    EMOTION_NAMES,
    FaceEmotionModel,
    VoiceEmotionModel,
    compute_prosody,
    load_audio,
)

logger = logging.getLogger(__name__)

# Upload limits
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_VIDEO_BYTES = 30 * 1024 * 1024  # 30 MB

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def validate_media_upload(file_bytes: bytes, filename: str, media_type: str) -> None:
    """Validate file extension and size limits."""
    ext = os.path.splitext(filename.lower())[1]
    size = len(file_bytes)

    if media_type == "audio":
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}")
        if size > MAX_AUDIO_BYTES:
            raise ValueError(f"Audio file exceeds maximum size limit of {MAX_AUDIO_BYTES // (1024*1024)}MB.")

    elif media_type == "video":
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}")
        if size > MAX_VIDEO_BYTES:
            raise ValueError(f"Video file exceeds maximum size limit of {MAX_VIDEO_BYTES // (1024*1024)}MB.")

    else:
        raise ValueError(f"Unknown media type: {media_type}")


def process_audio_file(file_bytes: bytes, filename: str, target_dim: int = 256) -> Dict:
    """
    Decodes an audio upload and runs the voice emotion model.
    feature_vector holds 7-class probabilities, or None when no usable audio exists.
    """
    validate_media_upload(file_bytes, filename, "audio")
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        wave_data = load_audio(tmp_path)
        prosody = compute_prosody(wave_data)
        return {
            "feature_vector": VoiceEmotionModel.predict(wave_data),
            "prosody": {
                **(prosody or {}),
                "duration_sec": round(len(wave_data) / 16000.0, 2) if wave_data is not None else 0.0,
                "status": "extracted" if prosody else "unavailable",
            },
        }

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary audio file {tmp_path}: {e}")


def process_video_file(file_bytes: bytes, filename: str, target_dim: int = 256) -> Dict:
    """
    Samples frames, finds faces and runs the facial emotion model.
    feature_vector holds 7-class probabilities, or None when no face is found.
    """
    validate_media_upload(file_bytes, filename, "video")
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        frames = []
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            for idx in np.linspace(0, total_frames - 1, min(8, total_frames), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
        cap.release()

        crops = [crop for crop in map(FaceEmotionModel.detect_largest_face, frames) if crop is not None]
        probabilities = None
        gradcam_base64 = None
        if crops:
            rows = FaceEmotionModel.predict_crops(crops)
            probabilities = rows.mean(axis=0)
            emotion = EMOTION_NAMES[int(np.argmax(probabilities))]
            best = int(np.argmax(rows[:, int(np.argmax(probabilities))]))
            gradcam_base64 = FaceEmotionModel.gradcam(crops[best], emotion)

        return {
            "feature_vector": probabilities,
            "visual_summary": {
                "frames_analyzed": len(frames),
                "faces_detected": len(crops),
                "fps": round(fps, 1),
                "gradcam_image_url": gradcam_base64,
            },
        }

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary video file {tmp_path}: {e}")
