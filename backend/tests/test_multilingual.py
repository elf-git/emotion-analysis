"""
test_multilingual.py - Comprehensive Unit & Integration Test Suite for Multilingual Emotion Recognition
Tests:
  - Language detection across English, Hindi, Malayalam, and Tamil
  - Manual user language overrides
  - Indic to English translation with script and Unicode preservation
  - Dual-approach text emotion analysis (Approach A native script vs Approach B translation)
  - Multilingual speech processing with timestamps and language detection
  - Full-clip video analysis with tri-modal alignment and scene timelines
  - Multilingual evaluation dataset integrity and benchmark reports
  - FastAPI endpoint handlers and response contracts
"""
import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.multilingual.language_detector import LanguageDetector, SUPPORTED_LANGUAGES
from ml.multilingual.translator import IndicTranslator
from ml.multilingual.text_analyzer import MultilingualTextAnalyzer, EMOTION_NAMES
from ml.multilingual.speech_processor import MultilingualSpeechProcessor
from backend.services.video_multilingual_analyzer import MultilingualVideoAnalyzer
from backend.main import (
    get_multilingual_languages,
    detect_language,
    predict_multilingual_text,
    get_multilingual_benchmark,
    DetectLanguageRequest,
    MultilingualTextRequest,
)


def test_supported_languages_metadata():
    """Verify supported languages contains English, Hindi, Malayalam, and Tamil with correct scripts."""
    meta = LanguageDetector.get_supported_languages()
    assert set(meta["codes"]) == {"en", "hi", "ml", "ta"}
    lang_dict = {l["code"]: l for l in meta["languages"]}
    assert lang_dict["hi"]["script"] == "Devanagari"
    assert lang_dict["ml"]["script"] == "Malayalam"
    assert lang_dict["ta"]["script"] == "Tamil"
    assert lang_dict["en"]["script"] == "Latin"


def test_language_detection_native_scripts():
    """Verify native script detection for Devanagari, Malayalam, Tamil, and English."""
    cases = [
        ("नमस्ते दुनिया, आप कैसे हैं?", "hi", "Hindi"),
        ("നമസ്കാരം സുഹൃത്തേ, സുഖമാണോ?", "ml", "Malayalam"),
        ("வணக்கம் நண்பா, எப்படி இருக்கிறீர்கள்?", "ta", "Tamil"),
        ("Hello world, this is a test in English.", "en", "English"),
    ]
    for text, expected_code, expected_name in cases:
        res = LanguageDetector.detect(text)
        assert res["language_code"] == expected_code
        assert res["language_name"] == expected_name
        assert res["confidence"] >= 0.70
        assert res["is_override"] is False


def test_language_detection_manual_override():
    """Verify user manual override is respected even if text script differs."""
    res = LanguageDetector.detect("Hello my friend", override_language="hi")
    assert res["language_code"] == "hi"
    assert res["language_name"] == "Hindi"
    assert res["is_override"] is True
    assert res["confidence"] == 1.0


def test_indic_translation_preservation():
    """Verify translation produces English while strictly preserving original script and text."""
    cases = [
        ("मैं बहुत खुश हूँ", "hi", "happy"),
        ("എനിക്ക് വളരെ സന്തോഷമുണ്ട്", "ml", "happy"),
        ("நான் ரொம்ப சந்தோஷமா இருக்கேன்", "ta", "happy"),
    ]
    for orig_text, lang, exp_keyword in cases:
        res = IndicTranslator.translate_to_english(orig_text, lang)
        assert res["original_text"] == orig_text
        assert res["source_language"] == lang
        assert res["is_translated"] is True
        assert exp_keyword in res["translated_text"].lower()


def test_text_analysis_approach_a_and_b():
    """Verify dual-approach text emotion analysis runs both native model and translation pipeline."""
    res = MultilingualTextAnalyzer.analyze("मैं आज बहुत खुश हूँ, हमारा सपना सच हो गया!", override_language="hi")
    assert res["detected_language"]["language_code"] == "hi"
    assert res["primary_prediction"] == "joy"

    # Approach A: Direct Multilingual
    app_a = res["approach_a"]
    assert app_a["evaluated_on_native_script"] is True
    assert app_a["predicted_emotion"] == "joy"
    assert len(app_a["token_attributions"]) > 0

    # Approach B: Translation-Based
    app_b = res["approach_b"]
    assert app_b["evaluated_on_translated_english"] is True
    assert "translated_text" in app_b

    # Comparison
    comp = res["comparison"]
    assert "agreed" in comp
    assert comp["consensus_emotion"] == "joy"


def test_text_analysis_malayalam_script():
    """Verify Malayalam text analysis handles Malayalam Unicode script properly."""
    res = MultilingualTextAnalyzer.analyze("എനിക്ക് വളരെ സങ്കടമുണ്ട്, ഞാൻ പൂർണ്ണമായും ഒറ്റപ്പെട്ടുപോയി.", override_language="ml")
    assert res["detected_language"]["language_code"] == "ml"
    assert res["primary_prediction"] == "sadness"
    assert res["approach_a"]["predicted_emotion"] == "sadness"


def test_text_analysis_tamil_script():
    """Verify Tamil text analysis handles Tamil Unicode script properly."""
    res = MultilingualTextAnalyzer.analyze("எனக்கு பயமாக இருக்கிறது, அங்கே ஏதோ பயங்கரமான ஆபத்து காத்திருக்கிறது!", override_language="ta")
    assert res["detected_language"]["language_code"] == "ta"
    assert res["primary_prediction"] == "fear"
    assert res["approach_a"]["predicted_emotion"] == "fear"


def test_text_analysis_translation_disabled():
    """Verify user can toggle translation off, running exclusively Approach A."""
    res = MultilingualTextAnalyzer.analyze("मुझे बहुत गुस्सा आ रहा है!", override_language="hi", allow_translation=False)
    assert res["approach_b"]["enabled"] is False
    assert res["primary_prediction"] == "anger"


def test_speech_processor_transcription():
    """Verify speech processor returns timestamps, detected language, and segments."""
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    res = MultilingualSpeechProcessor.transcribe(wav_bytes, language_override="hi", translate_to_english=True)
    assert "detected_language" in res
    assert res["detected_language"]["language_code"] == "hi"
    assert "original_transcript" in res
    assert "translated_transcript" in res
    assert len(res["segments"]) >= 1
    assert "start" in res["segments"][0]
    assert "end" in res["segments"][0]


def _blank_video_bytes() -> bytes:
    """A real two-second video with no faces and no audio."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "blank.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
        for _ in range(20):
            writer.write(np.full((120, 160, 3), 40, dtype=np.uint8))
        writer.release()
        with open(path, "rb") as f:
            return f.read()


def test_full_clip_video_analysis():
    res = MultilingualVideoAnalyzer.analyze_clip(_blank_video_bytes(), filename="clip.mp4", language_override="hi")
    assert res["status"] == "success"
    assert "scene_summary" in res
    assert "timeline" in res
    assert len(res["timeline"]) >= 1

    first_segment = res["timeline"][0]
    assert "timestamp" in first_segment
    assert "fused_emotion" in first_segment
    assert "modality_predictions" in first_segment
    assert "text" in first_segment["modality_predictions"]
    assert "acoustic" in first_segment["modality_predictions"]
    assert "visual" in first_segment["modality_predictions"]
    assert "visual_gradcam_image" in first_segment


def test_evaluation_datasets_integrity():
    """Verify all 4 evaluation datasets exist, have balanced classes, and valid JSONL formatting."""
    eval_dir = PROJECT_ROOT / "ml" / "data" / "multilingual_eval"
    for code in ["en", "hi", "ml", "ta"]:
        p = eval_dir / f"{code if code != 'en' else 'english' if not (eval_dir / 'en_eval.jsonl').exists() else 'en'}_eval.jsonl"
        # Check standard file name
        candidates = [eval_dir / f"{code}_eval.jsonl", eval_dir / f"{'hindi' if code=='hi' else 'malayalam' if code=='ml' else 'tamil' if code=='ta' else 'english'}_eval.jsonl"]
        file_path = next((f for f in candidates if f.exists()), None)
        assert file_path is not None, f"Dataset for {code} must exist"

        lines = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) >= 14, f"Dataset {code} should have at least 14 samples"
        emotions = set(item["emotion"] for item in lines)
        assert emotions == set(EMOTION_NAMES), f"Dataset {code} must cover all 7 classes"


def test_fastapi_multilingual_handlers():
    """Verify FastAPI route handlers directly as synchronous functions."""
    # 1. Languages endpoint
    langs = get_multilingual_languages()
    assert set(langs["codes"]) == {"en", "hi", "ml", "ta"}

    # 2. Detect language endpoint
    det = detect_language(DetectLanguageRequest(text="എനിക്ക് സന്തോഷമുണ്ട്"))
    assert det["language_code"] == "ml"

    # 3. Predict text endpoint
    pred = predict_multilingual_text(MultilingualTextRequest(text="मैं बहुत खुश हूँ!"))
    assert pred["primary_prediction"] == "joy"

    # 4. Benchmark endpoint
    bm = get_multilingual_benchmark()
    for code in ["en", "hi", "ml", "ta"]:
        assert code in bm



