"""
test_api.py - Automated Test Suite for FastAPI Backend
Tests endpoints, validation, media processing, uncertainty quantification, and error handling.
"""
import io
import sys
import wave
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    health_check,
    predict_dialogue,
    get_benchmark_results,
    get_sample_dialogues,
    DialogueRequest,
    DialogueTurn,
    engine,
)
from backend.services.media_processor import process_audio_file, process_video_file
from ml.pretrained_models import fuse


def test_health_endpoint():
    """Verify health endpoint returns 200 and model status."""
    data = health_check()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "supported_emotions" in data
    assert len(data["supported_emotions"]) == 7


def test_predict_text_only():
    """Verify multimodal endpoint works with text-only input."""
    data = engine.predict_single(text="I am so incredibly happy and proud of what we achieved today!")
    assert "predicted_emotion" in data
    assert "confidence_pct" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 7
    assert "modality_contributions" in data
    assert data["modality_contributions"]["text_pct"] > 0
    assert "uncertainty" in data
    assert "entropy" in data["uncertainty"]
    assert "explainability" in data
    assert len(data["explainability"]["token_attributions"]) > 0


def test_predict_empty_input_fails():
    """Verify empty input raises ValueError."""
    with pytest.raises(ValueError):
        engine.predict_single(text=None, audio_features=None, video_features=None)


def _tone_wav_bytes() -> bytes:
    """One second of a 220 Hz tone as 16 kHz mono WAV."""
    samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_predict_with_audio_upload():
    """Verify audio is decoded, measured and passed to the voice model."""
    proc_a = process_audio_file(_tone_wav_bytes(), "sample.wav")
    assert proc_a["feature_vector"].shape == (7,)
    assert proc_a["prosody"]["pitch_hz"] == pytest.approx(220, abs=5)

    res = engine.predict_single(
        text="Listen to how exciting this sounds!",
        audio_features=proc_a["feature_vector"],
        audio_summary=proc_a["prosody"],
    )
    assert res["active_modalities"]["audio"] is True
    assert res["explainability"]["prosody_summary"] is not None


def test_empty_audio_is_reported_as_missing():
    """Verify unusable audio gives no prediction instead of an invented one."""
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    proc_a = process_audio_file(wav_bytes, "empty.wav")
    assert proc_a["feature_vector"] is None
    assert proc_a["prosody"]["status"] == "unavailable"


def test_video_without_face_is_reported_as_missing():
    """Verify a video with no visible face gives no visual prediction."""
    video_bytes = b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00mp42mp41" + b"\x00" * 2000
    proc_v = process_video_file(video_bytes, "clip.mp4")
    assert proc_v["feature_vector"] is None
    assert proc_v["visual_summary"]["faces_detected"] == 0
    assert proc_v["visual_summary"]["gradcam_image_url"] is None


@pytest.mark.parametrize("text,expected", [
    ("I am so happy today, this is great", "joy"),
    ("I hate you, I am furious", "anger"),
    ("My dog died and I feel empty", "sadness"),
])
def test_text_prediction_reads_meaning(text, expected):
    """Verify predictions come from the model, including sentences with no keyword."""
    assert engine.predict_single(text=text)["predicted_emotion"] == expected


def test_prediction_is_repeatable():
    """Verify the same input always gives the same result."""
    first = engine.predict_single(text="I can't believe you did that!")
    second = engine.predict_single(text="I can't believe you did that!")
    assert first["probabilities"] == second["probabilities"]


def test_fusion_uses_only_present_modalities():
    """Verify missing modalities are skipped and no evidence gives no prediction."""
    anger = np.zeros(7, dtype=np.float32)
    anger[6] = 1.0
    result = fuse({"text": None, "audio": None, "visual": anger})
    assert result["predicted_emotion"] == "anger"
    assert result["confidence"] == pytest.approx(1.0)
    assert result["contributions"]["video_pct"] == 100.0
    assert fuse({"text": None, "audio": None, "visual": None}) is None


def test_dialogue_timeline():
    """Verify multi-turn dialogue emotion timeline."""
    req = DialogueRequest(
        dialogue=[
            DialogueTurn(speaker="Ross", text="Hi guys, I have some terrible news."),
            DialogueTurn(speaker="Rachel", text="Oh no Ross, what happened? Are you okay?"),
            DialogueTurn(speaker="Joey", text="Wait wait wait, does this mean pizza is cancelled?!"),
        ]
    )
    data = predict_dialogue(req)
    assert data["total_turns"] == 3
    assert len(data["timeline"]) == 3
    assert "conversational_tone" in data
    assert "emotion_distribution" in data


def test_benchmark_endpoint():
    """Verify official MELD benchmark data is returned."""
    data = get_benchmark_results()
    assert "benchmark" in data
    assert len(data["benchmark"]) >= 8


def test_sample_dialogues_endpoint():
    """Verify sample dialogues are available for instant UI testing."""
    data = get_sample_dialogues()
    assert "samples" in data
    assert len(data["samples"]) >= 3
