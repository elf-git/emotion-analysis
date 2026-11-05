"""
test_api.py - Automated Test Suite for FastAPI Backend
Tests endpoints, validation, media processing, uncertainty quantification, and error handling.
"""
import io
import sys
from pathlib import Path
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


def test_predict_with_audio_upload():
    """Verify audio file upload processing and prosody summary extraction."""
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    proc_a = process_audio_file(wav_bytes, "sample.wav", target_dim=engine.feature_dim)
    assert proc_a["feature_vector"] is not None
    assert proc_a["prosody"] is not None

    res = engine.predict_single(
        text="Listen to how exciting this sounds!",
        audio_features=proc_a["feature_vector"],
        audio_summary=proc_a["prosody"],
    )
    assert res["active_modalities"]["audio"] is True
    assert res["explainability"]["prosody_summary"] is not None


def test_predict_with_video_upload():
    """Verify video upload processing produces Grad-CAM visualization."""
    video_bytes = b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00mp42mp41" + b"\x00" * 2000
    proc_v = process_video_file(video_bytes, "clip.mp4", target_dim=engine.feature_dim)
    assert proc_v["feature_vector"] is not None
    assert proc_v["visual_summary"] is not None
    assert "gradcam_image_url" in proc_v["visual_summary"]

    res = engine.predict_single(
        text="Look at my expression right now!",
        video_features=proc_v["feature_vector"],
        video_summary=proc_v["visual_summary"],
    )
    assert res["active_modalities"]["video"] is True
    assert res["explainability"]["visual_summary"] is not None


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
