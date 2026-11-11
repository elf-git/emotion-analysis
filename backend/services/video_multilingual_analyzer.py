"""
video_multilingual_analyzer.py - Full-Clip Multilingual Video & Scene Emotion Analysis
Coordinates audio extraction, multilingual speech recognition, timestamp alignment,
modality feature extraction (Text, Audio, Video), tri-modal fusion, and scene aggregation.
"""
import os
import cv2
import tempfile
import subprocess
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from backend.services.media_processor import (
    validate_media_upload,
    process_audio_file,
    process_video_file,
    MAX_VIDEO_BYTES,
)
from backend.services.inference_service import (
    compute_uncertainty_metrics,
    EMOTION_NAMES,
    EMOTION_MAP,
    ID_TO_EMOTION,
    EMOTION_COLORS,
    EMOTION_VALENCE,
)
from ml.multilingual.language_detector import LanguageDetector, SUPPORTED_LANGUAGES
from ml.multilingual.speech_processor import MultilingualSpeechProcessor, NO_SPEECH_TEXT, has_speech
from ml.multilingual.text_analyzer import MultilingualTextAnalyzer
from ml.pretrained_models import (
    MODEL_INFO,
    SAMPLE_RATE,
    FaceEmotionModel,
    VoiceEmotionModel,
    compute_prosody,
    fuse,
    load_audio,
)

logger = logging.getLogger(__name__)

UNKNOWN_EMOTION = "unknown"
UNKNOWN_COLOR = "#52525b"
MAX_FRAMES_PER_SEGMENT = 5


class MultilingualVideoAnalyzer:
    """
    Analyzes full video clips with multilingual speech, facial dynamics,
    and acoustic prosody to produce a chronological scene emotion timeline.
    """

    @classmethod
    def analyze_clip(
        cls,
        video_bytes: bytes,
        filename: str = "video.mp4",
        language_override: Optional[str] = None,
        translate_to_english: bool = True,
        frame_sampling_interval_sec: float = 1.0,
        segment_interval_sec: Optional[float] = None,
        max_segments: int = 25,
    ) -> Dict[str, Any]:
        """
        Processes an uploaded video clip end-to-end:
        1. Validates video size & format.
        2. Extracts audio track.
        3. Runs multilingual speech transcription (Whisper) with timestamps.
        4. Partitions clip into dialogue or sliding-window intervals (configurable).
        5. Extracts facial frames (Grad-CAM) and acoustic features aligned to each segment.
        6. Computes Approach A (native script) and Approach B (translation) text emotions.
        7. Fuses modalities dynamically per turn and aggregates complete scene dynamics.
        """
        validate_media_upload(video_bytes, filename, "video")
        suffix = os.path.splitext(filename)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as vid_tmp:
            vid_tmp.write(video_bytes)
            video_path = vid_tmp.name

        audio_path = video_path + ".wav"

        try:
            # 1. Extract audio from video via ffmpeg
            audio_extracted = cls._extract_audio(video_path, audio_path)

            # 2. Transcribe speech using Multilingual Whisper
            wave_data = None
            if audio_extracted and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                wave_data = load_audio(audio_path)
                transcription_res = MultilingualSpeechProcessor.transcribe(
                    audio_path,
                    filename=os.path.basename(audio_path),
                    language_override=language_override,
                    translate_to_english=translate_to_english,
                )
            else:
                transcription_res = MultilingualSpeechProcessor._fallback_transcription(
                    audio_path, language_override, translate_to_english
                )

            detected_lang = transcription_res["detected_language"]
            lang_code = detected_lang["language_code"]
            raw_segments = transcription_res.get("segments", [])

            # 3. Analyze Video Metadata (FPS, duration, frames)
            video_meta = cls._get_video_metadata(video_path)
            if video_meta.get("total_frames", 0) == 0 and not audio_extracted:
                raise ValueError("This file has no readable video or audio.")
            total_duration = max(video_meta.get("duration", 2.0), transcription_res.get("total_duration_sec", 2.0))

            # 4. Segment Partitioning (Speech-aligned or sliding-window intervals)
            segments = []
            if segment_interval_sec and segment_interval_sec > 0.5:
                # User specified sliding-window segment intervals
                cur = 0.0
                while cur < total_duration and len(segments) < max_segments:
                    end_t = min(total_duration, cur + segment_interval_sec)
                    # Find matching dialogue text in this window
                    overlapping = [
                        s for s in raw_segments
                        if has_speech(s.get("text")) and s.get("start", 0) < end_t and s.get("end", total_duration) > cur
                    ]
                    win_text = " ".join(s["text"] for s in overlapping).strip() or NO_SPEECH_TEXT
                    win_translated = " ".join(s.get("translated_text", s["text"]) for s in overlapping).strip() or NO_SPEECH_TEXT
                    segments.append({
                        "start": round(cur, 2),
                        "end": round(end_t, 2),
                        "duration": round(end_t - cur, 2),
                        "text": win_text,
                        "translated_text": win_translated,
                    })
                    cur = end_t
            elif raw_segments:
                segments = raw_segments[:max_segments]
            else:
                segments = [{
                    "start": 0.0,
                    "end": round(total_duration, 2),
                    "duration": round(total_duration, 2),
                    "text": NO_SPEECH_TEXT,
                    "translated_text": NO_SPEECH_TEXT,
                    "confidence": 0.0,
                }]

            # 5. Segment-level Multi-Modal Alignment & Prediction
            timeline = []
            emotion_counts = {emo: 0 for emo in EMOTION_NAMES}

            for idx, seg in enumerate(segments):
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", total_duration)
                seg_native_text = seg.get("text", "")
                seg_translated_text = seg.get("translated_text", seg_native_text)

                # A. Text modality: only real recognized speech is scored
                text_analysis = None
                text_probs = None
                segment_has_speech = has_speech(seg_native_text)
                if segment_has_speech and detected_lang.get("supported", True):
                    text_analysis = MultilingualTextAnalyzer.analyze(
                        seg_native_text,
                        override_language=lang_code,
                        allow_translation=translate_to_english,
                        translated_text=seg_translated_text if has_speech(seg_translated_text) else None,
                    )
                elif segment_has_speech and has_speech(seg_translated_text) and seg_translated_text != seg_native_text:
                    # Other languages are scored through their English translation only
                    text_analysis = MultilingualTextAnalyzer.analyze(seg_translated_text, override_language="en")
                if text_analysis:
                    text_probs = text_analysis["primary_probabilities"]

                # B. Visual modality: faces in this time window
                visual_res = cls._extract_segment_visual(video_path, seg_start, seg_end, frame_sampling_interval_sec)

                # C. Acoustic modality: the voice model is only used where speech was
                # recognized, because it gives confident nonsense on music and noise
                acoustic_res = cls._extract_segment_acoustic(wave_data, seg_start, seg_end, segment_has_speech)

                # D. Weighted late fusion of the modalities that have evidence
                fusion = fuse({
                    "text": text_probs,
                    "audio": acoustic_res["probabilities"],
                    "visual": visual_res["probabilities"],
                })

                if fusion is None:
                    fused_probs = np.ones(len(EMOTION_NAMES), dtype=np.float32) / len(EMOTION_NAMES)
                    fused_emo = UNKNOWN_EMOTION
                    fused_conf = 0.0
                    contributions = {"text_pct": 0.0, "audio_pct": 0.0, "video_pct": 0.0, "dominant_modality": "None"}
                    explanation = "No speech, usable voice or visible face in this segment, so no emotion was estimated."
                else:
                    fused_probs = fusion["probabilities"]
                    fused_emo = fusion["predicted_emotion"]
                    fused_conf = fusion["confidence"]
                    contributions = fusion["contributions"]
                    emotion_counts[fused_emo] += 1
                    used = [
                        label for label, probs in (
                            ("text", text_probs),
                            ("voice", acoustic_res["probabilities"]),
                            ("face", visual_res["probabilities"]),
                        ) if probs is not None
                    ]
                    explanation = (
                        f"{fused_emo.upper()} ({round(fused_conf * 100.0, 1)}%) from late fusion of {', '.join(used)}. "
                        f"Strongest support: {contributions['dominant_modality']}."
                    )

                uncertainty = compute_uncertainty_metrics(fused_probs)
                fused_color = EMOTION_COLORS.get(fused_emo, UNKNOWN_COLOR)
                fused_valence = EMOTION_VALENCE.get(fused_emo, 0.0)
                fused_arousal = round(float(np.clip(0.5 + 0.35 * (fused_conf - 0.5) + (0.2 if fused_emo in ["anger", "surprise", "fear", "joy"] else -0.2), 0.1, 0.98)), 2)

                text_emo = text_analysis["primary_prediction"] if text_analysis else UNKNOWN_EMOTION
                text_conf = text_analysis["primary_confidence"] if text_analysis else 0.0
                approach_a = text_analysis["approach_a"] if text_analysis else {}
                approach_b = text_analysis["approach_b"] if text_analysis else {}
                acoustic_emo = acoustic_res["predicted_emotion"]
                acoustic_conf = acoustic_res["confidence"]
                visual_emo = visual_res["predicted_emotion"]
                visual_conf = visual_res["confidence"]
                prosody = acoustic_res["prosody"] or {}

                timeline_segment = {
                    "segment_id": idx + 1,
                    "segment_index": idx,
                    "timestamp": f"{seg_start:.1f}s - {seg_end:.1f}s",
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "start_time_sec": seg_start,
                    "end_time_sec": seg_end,
                    "timestamp_label": f"{int(seg_start//60):02d}:{seg_start%60:04.1f} - {int(seg_end//60):02d}:{seg_end%60:04.1f}",
                    "original_transcript": seg_native_text,
                    "translated_transcript": seg_translated_text,
                    "transcript": {
                        "original_text": seg_native_text,
                        "language": detected_lang["language_name"],
                        "language_code": lang_code,
                        "translated_text": seg_translated_text,
                    },
                    "language": detected_lang["language_name"],
                    "language_code": lang_code,
                    "fused_emotion": fused_emo,
                    "fused_confidence": round(fused_conf, 4),
                    "fused_confidence_pct": round(fused_conf * 100.0, 1),
                    "fused_prediction": {
                        "predicted_emotion": fused_emo,
                        "emotion_color": fused_color,
                        "confidence_pct": round(fused_conf * 100.0, 1),
                        "valence": fused_valence,
                        "arousal": fused_arousal,
                    },
                    "valence": fused_valence,
                    "color": fused_color,
                    "modality_predictions": {
                        "text": {
                            "predicted_emotion": text_emo,
                            "confidence": round(text_conf, 4),
                            "confidence_pct": round(text_conf * 100.0, 1),
                            "approach_a_multilingual": approach_a.get("predicted_emotion", text_emo),
                            "approach_b_translated": approach_b.get("predicted_emotion", text_emo),
                        },
                        "audio": {
                            "predicted_emotion": acoustic_emo,
                            "confidence": round(acoustic_conf, 4),
                            "confidence_pct": round(acoustic_conf * 100.0, 1),
                            "pitch_hz": prosody.get("pitch_hz"),
                            "rms_energy": prosody.get("rms_energy"),
                        },
                        "acoustic": {
                            "emotion": acoustic_emo,
                            "confidence": round(acoustic_conf, 4),
                            "predicted_emotion": acoustic_emo,
                            "confidence_pct": round(acoustic_conf * 100.0, 1),
                            "pitch_hz": prosody.get("pitch_hz"),
                            "rms_energy": prosody.get("rms_energy"),
                        },
                        "visual": {
                            "predicted_emotion": visual_emo,
                            "confidence": round(visual_conf, 4),
                            "confidence_pct": round(visual_conf * 100.0, 1),
                            "gradcam_available": visual_res["gradcam_base64"] is not None,
                            "gradcam_url": visual_res["gradcam_base64"],
                            "faces_detected": visual_res["faces_detected"],
                            "frames_sampled": visual_res["frames_sampled"],
                        },
                    },
                    "modality_contributions": contributions,
                    "uncertainty": uncertainty,
                    "explanation": explanation,
                    "text_details": {
                        "approach_a": {
                            "predicted_emotion": approach_a.get("predicted_emotion", text_emo),
                            "confidence": approach_a.get("confidence", 0.0),
                        },
                        "approach_b": {
                            "predicted_emotion": approach_b.get("predicted_emotion", text_emo),
                            "confidence": approach_b.get("confidence", text_conf),
                        },
                        "token_attributions": approach_a.get("token_attributions", [])[:5],
                    },
                    "visual_gradcam_image": visual_res.get("gradcam_base64"),
                    "acoustic_prosody": acoustic_res["prosody"],
                }
                timeline.append(timeline_segment)

            # 6. Emotion Transitions
            transitions = []
            for i in range(len(timeline) - 1):
                cur_emo = timeline[i]["fused_emotion"]
                next_emo = timeline[i + 1]["fused_emotion"]
                if cur_emo != next_emo:
                    transitions.append({
                        "from_turn": i + 1,
                        "to_turn": i + 2,
                        "transition": f"{cur_emo} -> {next_emo}",
                        "timestamp": timeline[i + 1]["timestamp"],
                        "note": f"Shift from {cur_emo.upper()} to {next_emo.upper()} observed in scene.",
                    })

            # 7. Scene-level Emotion Distribution
            total_segs = max(1, len(timeline))
            scene_distribution = {
                emo: {
                    "count": emotion_counts[emo],
                    "percentage": round((emotion_counts[emo] / total_segs) * 100.0, 1),
                    "color": EMOTION_COLORS[emo],
                }
                for emo in EMOTION_NAMES
            }

            dominant_emotion = max(emotion_counts, key=lambda k: emotion_counts[k])
            top_count = emotion_counts[dominant_emotion]
            if top_count == 0:
                dominant_emotion = UNKNOWN_EMOTION
            elif sum(count == top_count for count in emotion_counts.values()) > 1:
                dominant_emotion = "mixed"
            scored = [s for s in timeline if s["fused_emotion"] != UNKNOWN_EMOTION]
            avg_val = round(float(np.mean([s["fused_prediction"]["valence"] for s in scored])), 2) if scored else 0.0
            avg_aro = round(float(np.mean([s["fused_prediction"]["arousal"] for s in scored])), 2) if scored else 0.0
            trans_list = [s["fused_prediction"]["predicted_emotion"] for s in timeline]

            return {
                "status": "success",
                "is_live_inference": True,
                "inference_source": "pretrained_model_pipeline",
                "models": MODEL_INFO,
                "filename": filename,
                "video_metadata": video_meta,
                "scene_metadata": {
                    "filename": filename,
                    "duration_sec": round(total_duration, 2),
                    "fps": video_meta.get("fps", 24.0),
                    "total_frames_sampled": video_meta.get("total_frames", 0),
                    "audio_extracted": audio_extracted,
                    "detected_language": detected_lang["language_name"],
                    "language_code": lang_code,
                },
                "speech_transcription": {
                    "original_transcript": transcription_res.get("original_transcript", ""),
                    "translated_transcript": transcription_res.get("translated_transcript", ""),
                    "detected_language": detected_lang,
                    "is_translated": transcription_res.get("is_translated", False),
                },
                "scene_summary": {
                    "total_segments": len(timeline),
                    "total_duration_sec": round(total_duration, 2),
                    "dominant_emotion": dominant_emotion,
                    "dominant_emotion_color": EMOTION_COLORS.get(dominant_emotion, UNKNOWN_COLOR),
                    "average_valence": avg_val,
                    "average_arousal": avg_aro,
                    "emotion_transitions": trans_list,
                    "emotion_transitions_count": len(transitions),
                    "emotion_distribution": scene_distribution,
                },
                "segments": timeline,
                "timeline": timeline,
                "emotion_transitions": transitions,
            }

        finally:
            if os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception:
                    pass
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    @classmethod
    def _extract_audio(cls, video_path: str, audio_path: str) -> bool:
        """Extracts mono 16kHz WAV audio from video via ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            return res.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        except Exception as e:
            logger.warning(f"ffmpeg audio extraction error: {e}")
            return False

    @classmethod
    def _get_video_metadata(cls, video_path: str) -> Dict[str, Any]:
        """Reads FPS, frame dimensions, and clip duration using OpenCV."""
        meta = {"fps": 24.0, "total_frames": 0, "duration": 2.0, "width": 640, "height": 360}
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
            duration = (total_frames / fps) if fps > 0 else 2.0
            cap.release()
            meta.update({
                "fps": round(float(fps), 2),
                "total_frames": total_frames,
                "duration": round(float(duration), 2),
                "width": width,
                "height": height,
            })
        except Exception as e:
            logger.warning(f"Could not read video metadata: {e}")
        return meta

    @classmethod
    def _extract_segment_visual(cls, video_path: str, start_sec: float, end_sec: float, sampling_sec: float) -> Dict[str, Any]:
        """Runs the face emotion model on frames sampled across a time window."""
        duration = max(0.0, end_sec - start_sec)
        count = int(np.clip(round(duration / max(0.25, sampling_sec)), 1, MAX_FRAMES_PER_SEGMENT))
        timestamps = np.linspace(start_sec, end_sec, count + 2)[1:-1]

        crops = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames_read = 0
        if fps > 0:
            for timestamp in timestamps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frames_read += 1
                crop = FaceEmotionModel.detect_largest_face(frame)
                if crop is not None:
                    crops.append(crop)
        cap.release()

        if not crops:
            return {
                "probabilities": None,
                "predicted_emotion": UNKNOWN_EMOTION,
                "confidence": 0.0,
                "gradcam_base64": None,
                "faces_detected": 0,
                "frames_sampled": frames_read,
            }

        rows = FaceEmotionModel.predict_crops(crops)
        probs = rows.mean(axis=0)
        pred_idx = int(np.argmax(probs))
        emotion = EMOTION_NAMES[pred_idx]
        return {
            "probabilities": probs,
            "predicted_emotion": emotion,
            "confidence": round(float(probs[pred_idx]), 4),
            "gradcam_base64": FaceEmotionModel.gradcam(crops[int(np.argmax(rows[:, pred_idx]))], emotion),
            "faces_detected": len(crops),
            "frames_sampled": frames_read,
        }

    @classmethod
    def _extract_segment_acoustic(
        cls, wave_data: Optional[np.ndarray], start_sec: float, end_sec: float, speech_present: bool
    ) -> Dict[str, Any]:
        """Runs the voice emotion model and measures prosody for a time window."""
        window = None
        if wave_data is not None:
            window = wave_data[int(start_sec * SAMPLE_RATE): int(end_sec * SAMPLE_RATE)]

        probs = VoiceEmotionModel.predict(window) if speech_present else None
        if probs is None:
            return {
                "probabilities": None,
                "predicted_emotion": UNKNOWN_EMOTION,
                "confidence": 0.0,
                "prosody": compute_prosody(window),
            }

        pred_idx = int(np.argmax(probs))
        return {
            "probabilities": probs,
            "predicted_emotion": EMOTION_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "prosody": compute_prosody(window),
        }
