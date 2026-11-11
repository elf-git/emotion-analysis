"""
speech_processor.py - Multilingual Speech Transcription & Timestamp Alignment
Uses faster-whisper / Whisper models to transcribe conversational speech
in English (en), Hindi (hi), Malayalam (ml), and Tamil (ta).
Preserves native transcripts, segment timestamps, speech language detection,
and optional direct English translation. Decoupled from emotion inference.
"""
import os
import io
import wave
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from ml.multilingual.language_detector import LanguageDetector, SUPPORTED_LANGUAGES
from ml.multilingual.translator import IndicTranslator

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent / ".model_cache")

# Shown when no speech is available. Text starting with "[" is never scored for emotion.
NO_SPEECH_TEXT = "[No speech recognized]"


def has_speech(text: Optional[str]) -> bool:
    """True when text is real recognized speech rather than a placeholder."""
    cleaned = (text or "").strip()
    return bool(cleaned) and not cleaned.startswith("[")


class MultilingualSpeechProcessor:
    """
    Decoupled service handling multilingual speech recognition, timestamp alignment,
    and audio language detection.
    """

    _whisper_model = None
    _model_size = "small"  # "tiny" misses too much film dialogue
    _is_initialized = False

    @classmethod
    def get_whisper_model(cls):
        """Lazy load faster-whisper model."""
        if cls._whisper_model is None and not cls._is_initialized:
            try:
                from faster_whisper import WhisperModel
                # Run on CPU with INT8 quantization for optimal speed and memory footprint
                cls._whisper_model = WhisperModel(
                    cls._model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    download_root=MODEL_CACHE_DIR,
                    local_files_only=os.path.isdir(
                        os.path.join(MODEL_CACHE_DIR, f"models--Systran--faster-whisper-{cls._model_size}")
                    ),
                )
                logger.info(f"Loaded faster-whisper '{cls._model_size}' model for multilingual transcription.")
            except Exception as e:
                logger.warning(f"Could not load faster-whisper model: {e}. Using fallback speech processor.")
                cls._whisper_model = None
            cls._is_initialized = True
        return cls._whisper_model

    @classmethod
    def _run_whisper(cls, model, audio_path: str, language: Optional[str], task: str, vad_filter: bool):
        """Runs one Whisper pass, dropping segments that look like hallucinated non-speech."""
        segments_gen, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=3,
            vad_filter=vad_filter,
        )
        kept = [
            seg for seg in segments_gen
            if seg.text.strip() and not (seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0)
        ]
        return kept, info

    @staticmethod
    def _overlapping_text(start: float, end: float, translated_segments) -> str:
        """Joins translated Whisper segments whose midpoint falls inside a time window."""
        return " ".join(
            seg.text.strip() for seg in translated_segments
            if start <= (seg.start + seg.end) / 2.0 <= end
        ).strip()

    @classmethod
    def transcribe(
        cls,
        audio_input: Union[str, bytes],
        filename: str = "audio.wav",
        language_override: Optional[str] = None,
        translate_to_english: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio into timestamped segments and detected language.

        Args:
            audio_input: File path (str) or raw bytes of audio file.
            filename: Original file name with extension.
            language_override: Optional forced language ('en', 'hi', 'ml', 'ta').
            translate_to_english: If True, provides translated English text alongside native transcript.

        Returns:
            Dictionary containing detected speech language, segments with timestamps,
            full native transcript, and optional English translation.
        """
        tmp_path = None
        if isinstance(audio_input, bytes):
            suffix = os.path.splitext(filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_input)
                tmp_path = tmp.name
            audio_path = tmp_path
        else:
            audio_path = str(audio_input)

        try:
            model = cls.get_whisper_model()
            lang_param = language_override.lower() if language_override else None
            if lang_param and lang_param not in SUPPORTED_LANGUAGES:
                lang_param = None

            if model is not None and os.path.exists(audio_path):
                try:
                    # 1. Native transcription. Voice activity detection can discard
                    # speech under film music, so retry without it before giving up.
                    whisper_segments, info = cls._run_whisper(model, audio_path, lang_param, "transcribe", True)
                    if not whisper_segments:
                        whisper_segments, info = cls._run_whisper(model, audio_path, lang_param, "transcribe", False)
                    detected_lang = info.language
                    lang_prob = float(info.language_probability)

                    # 2. Speech translation pass for non-English audio
                    translated_segments = []
                    if translate_to_english and detected_lang != "en" and whisper_segments:
                        translated_segments, _ = cls._run_whisper(model, audio_path, detected_lang, "translate", False)

                    segments = []
                    native_texts = []
                    for seg in whisper_segments:
                        seg_text = seg.text.strip()
                        native_texts.append(seg_text)
                        seg_data = {
                            "start": round(float(seg.start), 2),
                            "end": round(float(seg.end), 2),
                            "duration": round(float(seg.end - seg.start), 2),
                            "text": seg_text,
                            "confidence": round(float(np.exp(seg.avg_logprob)), 3),
                            "translated_text": seg_text,
                        }
                        if translated_segments:
                            seg_data["translated_text"] = (
                                cls._overlapping_text(seg.start, seg.end, translated_segments) or seg_text
                            )
                        segments.append(seg_data)

                    full_native_transcript = " ".join(native_texts).strip()

                    # No speech is reported as such, never invented
                    if not full_native_transcript:
                        full_native_transcript = NO_SPEECH_TEXT
                        segments = [{
                            "start": 0.0,
                            "end": round(info.duration, 2) if hasattr(info, "duration") else 1.0,
                            "duration": round(info.duration, 2) if hasattr(info, "duration") else 1.0,
                            "text": full_native_transcript,
                            "translated_text": full_native_transcript,
                            "confidence": 0.5,
                        }]

                    # Format language info
                    if language_override and language_override.lower() in SUPPORTED_LANGUAGES:
                        final_lang = language_override.lower()
                        is_override = True
                        conf = 1.0
                        supported = True
                    else:
                        supported = detected_lang in SUPPORTED_LANGUAGES
                        final_lang = detected_lang if supported else "en"
                        is_override = False
                        conf = lang_prob

                    lang_meta = SUPPORTED_LANGUAGES.get(final_lang, SUPPORTED_LANGUAGES["en"])

                    # Build full translation
                    if translated_segments:
                        full_trans = " ".join(seg.text.strip() for seg in translated_segments)
                    else:
                        full_trans = full_native_transcript

                    return {
                        "status": "success",
                        "detected_language": {
                            "language_code": final_lang,
                            "language_name": lang_meta["name"] if supported else f"Other ({detected_lang})",
                            "native_name": lang_meta["native_name"] if supported else detected_lang,
                            "confidence": round(conf, 4),
                            "is_override": is_override,
                            "supported": supported,
                        },
                        "original_transcript": full_native_transcript,
                        "translated_transcript": full_trans,
                        "is_translated": bool(translated_segments),
                        "total_duration_sec": round(float(info.duration), 2) if hasattr(info, "duration") else 0.0,
                        "segments": segments,
                    }

                except Exception as ex:
                    logger.warning(f"Whisper inference failed on {audio_path}: {ex}. Reporting no transcript.")

            return cls._fallback_transcription(audio_path, language_override, translate_to_english)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    @classmethod
    def _fallback_transcription(
        cls,
        audio_path: str,
        language_override: Optional[str],
        translate_to_english: bool,
    ) -> Dict[str, Any]:
        """Used when speech could not be processed. Reports that no transcript exists."""
        lang_code = (language_override or "en").lower()
        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = "en"
        meta = SUPPORTED_LANGUAGES[lang_code]

        return {
            "status": "no_transcript",
            "detected_language": {
                "language_code": lang_code,
                "language_name": meta["name"],
                "native_name": meta["native_name"],
                "confidence": 0.0,
                "is_override": bool(language_override),
            },
            "original_transcript": NO_SPEECH_TEXT,
            "translated_transcript": NO_SPEECH_TEXT,
            "is_translated": False,
            "total_duration_sec": 0.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.0,
                    "duration": 0.0,
                    "text": NO_SPEECH_TEXT,
                    "translated_text": NO_SPEECH_TEXT,
                    "confidence": 0.0,
                }
            ],
        }
