"""
test_groq_service.py - Unit tests for Groq Cloud LPU speech & cognitive reasoning services
"""
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.groq_service import GroqService, LANG_MAP


def test_groq_service_language_mapping():
    """Verify supported language map includes all major Indic and World cinema languages."""
    assert "hindi" in LANG_MAP
    assert "malayalam" in LANG_MAP
    assert "tamil" in LANG_MAP
    assert "telugu" in LANG_MAP
    assert "kannada" in LANG_MAP
    assert "english" in LANG_MAP
    assert LANG_MAP["hindi"][0] == "hi"
    assert LANG_MAP["malayalam"][0] == "ml"
    assert LANG_MAP["tamil"][0] == "ta"


def test_groq_service_availability_check():
    """Verify GroqService.is_available() executes safely without raising exceptions."""
    available = GroqService.is_available()
    assert isinstance(available, bool)


def test_groq_service_cognitive_explanation_fallback():
    """Verify GroqService.generate_cognitive_explanation produces valid academic output."""
    res = GroqService.generate_cognitive_explanation(
        movie_name="Kumbalangi Nights",
        dialogue_text="എന്താടാ ഇവിടെ നടക്കുന്നത്?",
        predicted_emotion="anger",
        text_prediction="anger",
        audio_prediction="anger",
        visual_prediction="anger",
        text_pct=36.0,
        audio_pct=34.0,
        video_pct=30.0,
    )
    assert isinstance(res, str)
    assert len(res) > 15
    assert "anger" in res.lower() or "attention" in res.lower() or "multimodal" in res.lower()


def test_groq_service_transcribe_missing_file_safe():
    """Verify transcribe_audio safely returns None for non-existent files without crashing."""
    res = GroqService.transcribe_audio("non_existent_audio_file_123456.wav")
    assert res is None
