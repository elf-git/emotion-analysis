"""
main.py - FastAPI Application for Multimodal Emotion Recognition & Explainability
Provides REST endpoints for multimodal prediction, dialogue timelines, explainability,
and benchmark results with safe media handling and upload limits.
"""
import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env environment variables securely
load_dotenv(PROJECT_ROOT / ".env")

from backend.services.inference_service import (
    MultimodalInferenceEngine,
    EMOTION_NAMES,
    EMOTION_COLORS,
)
from backend.services.media_processor import (
    process_audio_file,
    process_video_file,
    MAX_AUDIO_BYTES,
    MAX_VIDEO_BYTES,
)
from backend.services.youtube_downloader import YouTubeClipDownloader
from ml.pretrained_models import MODEL_INFO, warm_up
from ml.multilingual.language_detector import LanguageDetector, SUPPORTED_LANGUAGES
from ml.multilingual.text_analyzer import MultilingualTextAnalyzer
from ml.multilingual.speech_processor import MultilingualSpeechProcessor
from ml.multilingual.translator import IndicTranslator
from backend.services.video_multilingual_analyzer import MultilingualVideoAnalyzer
from backend.services.serp_scene_service import SerpMovieSceneService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Emotion Recognition & Communication Analysis API",
    description="Explainable Multimodal Emotion AI for Conversational Audio, Visual & Linguistic Dynamics (MELD Benchmark)",
    version="1.0.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = MultimodalInferenceEngine.get_instance()


def _warm_models() -> None:
    try:
        warm_up()
        MultilingualSpeechProcessor.get_whisper_model()
        logger.info("All models loaded and ready.")
    except Exception as e:
        logger.warning(f"Model warm-up failed, models will load on first use: {e}")


@app.on_event("startup")
def start_model_warm_up() -> None:
    """Loads models in the background so the server starts straight away."""
    threading.Thread(target=_warm_models, daemon=True).start()


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class DialogueTurn(BaseModel):
    speaker: str = Field(..., json_schema_extra={"example": "Chandler"})
    text: str = Field(..., json_schema_extra={"example": "Could I BE any more happy for you?!"})


class DialogueRequest(BaseModel):
    dialogue: List[DialogueTurn]


class DetectLanguageRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "എനിക്ക് വളരെ സന്തോഷമുണ്ട്"})
    override_language: Optional[str] = Field(None, json_schema_extra={"example": "ml"})


class MultilingualTextRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "मैं आज बहुत खुश हूँ!"})
    override_language: Optional[str] = Field(None, json_schema_extra={"example": "hi"})
    allow_translation: bool = Field(True, json_schema_extra={"example": True})


# -----------------------------------------------------------------------------
# Global Exception Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid Input", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check endpoint returning device and loaded artifact status."""
    return {
        "status": "healthy",
        "models_loaded": engine.is_loaded,
        "model_status": engine.model_status,
        "models": MODEL_INFO,
        "supported_emotions": EMOTION_NAMES,
        "upload_limits_mb": {
            "audio_max_mb": MAX_AUDIO_BYTES // (1024 * 1024),
            "video_max_mb": MAX_VIDEO_BYTES // (1024 * 1024),
        },
    }


@app.post("/api/predict/multimodal")
async def predict_multimodal(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    """
    Multimodal emotion prediction accepting any combination of Text, Audio, and Video.
    Extracts features, computes cross-modal attention, modality gate weights,
    and returns uncertainty information with explainability.
    """
    if not text and audio is None and video is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide at least one input modality (text string, audio file, or video file).",
        )

    audio_features, audio_summary = None, None
    video_features, video_summary = None, None

    # Process Audio Upload
    if audio is not None:
        try:
            audio_bytes = await audio.read()
            if len(audio_bytes) > MAX_AUDIO_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio file exceeds maximum size limit of {MAX_AUDIO_BYTES // (1024*1024)}MB.",
                )
            proc_a = process_audio_file(audio_bytes, audio.filename or "input.wav", target_dim=256)
            audio_features = proc_a["feature_vector"]
            audio_summary = proc_a["prosody"]
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    # Process Video Upload
    if video is not None:
        try:
            video_bytes = await video.read()
            if len(video_bytes) > MAX_VIDEO_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Video file exceeds maximum size limit of {MAX_VIDEO_BYTES // (1024*1024)}MB.",
                )
            proc_v = process_video_file(video_bytes, video.filename or "input.mp4", target_dim=256)
            video_features = proc_v["feature_vector"]
            video_summary = proc_v["visual_summary"]
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    result = engine.predict_single(
        text=text,
        audio_features=audio_features,
        video_features=video_features,
        audio_summary=audio_summary,
        video_summary=video_summary,
    )
    return result


@app.post("/api/predict/dialogue")
def predict_dialogue(request: DialogueRequest):
    """
    Analyzes an entire conversational dialogue sequence.
    Returns a chronological emotion timeline, valence dynamics, and speaker analysis.
    """
    if not request.dialogue:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dialogue turns list cannot be empty.")

    turns_dicts = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in request.dialogue]
    timeline_res = engine.predict_dialogue_timeline(turns_dicts)
    return timeline_res


def _load_smoke_test_results() -> Dict[str, Any]:
    """Reads the only measured results in the repository: short smoke-test runs."""
    results = {}
    for name in ("text", "audio", "video"):
        path = PROJECT_ROOT / "checkpoints" / f"{name}_model_results.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = data.get("test_metrics", {})
            results[name] = {
                "smoke_test": data.get("smoke_test"),
                "epochs": len(data.get("history", [])),
                "test_accuracy_pct": round(metrics.get("accuracy", 0.0) * 100.0, 1),
                "test_f1_weighted_pct": round(metrics.get("f1_weighted", 0.0) * 100.0, 1),
            }
    return results


@app.get("/api/benchmark")
def get_benchmark_results():
    """Returns the official MELD ablation and multimodal benchmark results."""
    ablation_data = [
        {"configuration": "Text Baseline (DistilBERT)", "modality": "Linguistic", "weights": "1.0 Text", "test_acc": 78.35, "macro_f1": 68.71, "weighted_f1": 79.82},
        {"configuration": "Audio Baseline (openSMILE)", "modality": "Acoustic", "weights": "1.0 Audio", "test_acc": 76.09, "macro_f1": 66.64, "weighted_f1": 77.60},
        {"configuration": "Video Baseline (Visual ResNet)", "modality": "Visual", "weights": "1.0 Video", "test_acc": 86.36, "macro_f1": 78.54, "weighted_f1": 87.07},
        {"configuration": "Text + Audio Late Fusion", "modality": "Text + Audio", "weights": "0.50 T + 0.50 A", "test_acc": 91.38, "macro_f1": 85.67, "weighted_f1": 91.65},
        {"configuration": "Text + Video Late Fusion", "modality": "Text + Video", "weights": "0.40 T + 0.60 V", "test_acc": 94.60, "macro_f1": 90.67, "weighted_f1": 94.72},
        {"configuration": "Audio + Video Late Fusion", "modality": "Audio + Video", "weights": "0.45 A + 0.55 V", "test_acc": 94.75, "macro_f1": 90.30, "weighted_f1": 94.89},
        {"configuration": "Tri-Modal Late Fusion (Equal)", "modality": "Text + Audio + Video", "weights": "0.33 T + 0.33 A + 0.34 V", "test_acc": 98.12, "macro_f1": 96.18, "weighted_f1": 98.15},
        {"configuration": "Tri-Modal Late Fusion (Optimized)", "modality": "Text + Audio + Video", "weights": "0.35 T + 0.27 A + 0.38 V", "test_acc": 98.08, "macro_f1": 96.14, "weighted_f1": 98.11},
        {"configuration": "Multimodal Gated Attention Fusion", "modality": "Text + Audio + Video", "weights": "Dynamic Modality Gate", "test_acc": 85.75, "macro_f1": 71.82, "weighted_f1": 85.96},
    ]
    return {
        "dataset": "MELD (Multimodal EmotionLines Dataset)",
        "split_sizes": {"train": 9989, "dev": 1109, "test": 2610},
        "benchmark": ablation_data,
        "verified": False,
        "warning": (
            "These figures are fixed values from the repository. No trained weights exist to reproduce them, "
            "and they do not describe the models this app runs."
        ),
        "saved_smoke_test_results": _load_smoke_test_results(),
    }


@app.get("/api/samples")
def get_sample_dialogues():
    """
    Returns curated multimodal conversational dialogues from MELD for quick demonstration.
    """
    samples = [
        {
            "id": "sample_joy",
            "title": "Ecstatic Life Event (Joy & Celebration)",
            "scenario": "Monica and Chandler celebrating exciting news with friends.",
            "dominant_emotion": "joy",
            "dialogue": [
                {"speaker": "Monica", "text": "Guess what just happened?! We actually got the apartment!"},
                {"speaker": "Chandler", "text": "Are you serious?! Oh my God, this is the best day of my entire life!"},
                {"speaker": "Joey", "text": "That is unbelievable! We gotta throw a party right now!"},
            ]
        },
        {
            "id": "sample_conflict",
            "title": "Heated Disagreement (Anger & Frustration)",
            "scenario": "Ross and Rachel in an emotional interpersonal argument.",
            "dominant_emotion": "anger",
            "dialogue": [
                {"speaker": "Rachel", "text": "I can't believe you did that behind my back!"},
                {"speaker": "Ross", "text": "I didn't do it to hurt you, why won't you just listen to me for one second?!"},
                {"speaker": "Rachel", "text": "Because every time you explain yourself, it just makes things ten times worse!"},
            ]
        },
        {
            "id": "sample_empathy",
            "title": "Vulnerable Moment (Sadness & Comfort)",
            "scenario": "Friends consoling each other through a painful breakup.",
            "dominant_emotion": "sadness",
            "dialogue": [
                {"speaker": "Phoebe", "text": "I don't think I can handle this pain anymore, I just feel completely alone."},
                {"speaker": "Monica", "text": "Hey, look at me. You are never alone as long as we are here, okay?"},
                {"speaker": "Phoebe", "text": "I know... it just hurts so much right now."},
            ]
        },
        {
            "id": "sample_sarcasm",
            "title": "Ambiguous / Sarcastic Dialogue (Cross-Modal Dissonance)",
            "scenario": "Chandler delivering a classic deadpan quip where words seem positive but vocal tone indicates sarcasm.",
            "dominant_emotion": "neutral",
            "dialogue": [
                {"speaker": "Joey", "text": "Hey Chandler, I just fixed the coffee table using only scotch tape!"},
                {"speaker": "Chandler", "text": "Well isn't that just a brilliant triumph of modern civil engineering."},
            ]
        }
    ]
    return {"samples": samples}


# -----------------------------------------------------------------------------
# Multilingual Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/multilingual/languages")
def get_multilingual_languages():
    """Returns supported languages (English, Hindi, Malayalam, Tamil) and metadata."""
    return LanguageDetector.get_supported_languages()


@app.post("/api/multilingual/detect-language")
def detect_language(request: DetectLanguageRequest):
    """Identifies language, script, confidence, and respects manual user overrides."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text cannot be empty.")
    return LanguageDetector.detect(request.text, override_language=request.override_language)


@app.post("/api/multilingual/predict/text")
def predict_multilingual_text(request: MultilingualTextRequest):
    """
    Evaluates text across both Approach A (Direct Multilingual Model on native Indic script)
    and Approach B (Translation via IndicTrans2 + English MELD DistilBERT).
    Returns comparative predictions, consensus, divergence, and token attributions.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text cannot be empty.")
    try:
        return MultilingualTextAnalyzer.analyze(
            request.text,
            override_language=request.override_language,
            allow_translation=request.allow_translation,
        )
    except Exception as e:
        logger.error(f"Error in multilingual text analysis: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/multilingual/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language_override: Optional[str] = Form(None),
    translate_to_english: bool = Form(True),
):
    """
    Multilingual speech-to-text using Whisper with automatic language detection,
    timestamped segmentation, and optional English translation.
    """
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum size limit of {MAX_AUDIO_BYTES // (1024*1024)}MB.",
        )
    try:
        return MultilingualSpeechProcessor.transcribe(
            audio_bytes,
            filename=audio.filename or "input.wav",
            language_override=language_override,
            translate_to_english=translate_to_english,
        )
    except Exception as e:
        logger.error(f"Error transcribing speech: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/multilingual/analyze/clip")
async def analyze_multilingual_video_clip(
    video: UploadFile = File(...),
    language_override: Optional[str] = Form(None),
    translate_to_english: bool = Form(True),
):
    """
    Comprehensive full-clip multilingual video analysis:
    Audio extraction -> Multilingual Whisper transcription -> Timestamped segment alignment
    -> Tri-modal prediction (Text Approach A/B, Audio prosody, Video Grad-CAM)
    -> Chronological scene timeline, emotion distribution, and transitions.
    """
    video_bytes = await video.read()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file exceeds maximum size limit of {MAX_VIDEO_BYTES // (1024*1024)}MB.",
        )
    try:
        return MultilingualVideoAnalyzer.analyze_clip(
            video_bytes,
            filename=video.filename or "video.mp4",
            language_override=language_override,
            translate_to_english=translate_to_english,
        )
    except Exception as e:
        logger.error(f"Error in multilingual full-clip video analysis: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/multilingual/benchmark")
def get_multilingual_benchmark():
    """Returns the empirical benchmark results comparing Approach A and Approach B across languages."""
    benchmark_file = PROJECT_ROOT / "exports" / "multilingual_benchmark_results.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark results not yet generated.")


# -----------------------------------------------------------------------------
# Movie Scene Discovery (SerpApi) & Full-Clip Scene Analysis Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/scenes/search")
def search_movie_scenes(
    movie: Optional[str] = None,
    scene_desc: Optional[str] = None,
    language: Optional[str] = "all",
    emotion: Optional[str] = "all",
    character: Optional[str] = None,
    page: int = 1,
    per_page: int = 8,
):
    """
    Discovers movie scene clips using SerpApi YouTube Search.
    Supports English, Hindi, Malayalam, and Tamil with emotion keywords,
    character queries, and resilient mock fallback.
    """
    return SerpMovieSceneService.search_scenes(
        movie_name=movie,
        scene_description=scene_desc,
        language=language,
        emotion=emotion,
        character_name=character,
        page=page,
        per_page=per_page,
    )


@app.get("/api/scenes/featured")
def get_featured_scenes():
    """Returns curated featured movie scenes across English, Hindi, Malayalam, and Tamil."""
    return SerpMovieSceneService.get_featured_scenes()


@app.post("/api/scenes/analyze-clip")
async def analyze_full_movie_scene_clip(
    video: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    language_override: Optional[str] = Form(None),
    translate_to_english: bool = Form(True),
    include_translation: Optional[bool] = Form(None),
    frame_sampling_interval_sec: float = Form(1.0),
    segment_interval_sec: Optional[float] = Form(None),
    max_segments: int = Form(25),
):
    """
    Performs full-clip multimodal emotion analysis on an authorized uploaded movie scene:
    1. Extracts audio & runs multilingual Whisper speech-to-text.
    2. Partitions clip into dialogue turns or sliding-window intervals.
    3. Analyzes facial frames for visual Grad-CAM saliency.
    4. Extracts acoustic prosody dynamics.
    5. Evaluates text via Approach A (native script) and Approach B (translation).
    6. Fuses modalities dynamically per turn and outputs chronological scene timeline.
    """
    target_upload = video or video_file
    if not target_upload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file provided. Expected 'video' or 'video_file' in multipart form-data.",
        )

    video_bytes = await target_upload.read()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file exceeds maximum size limit of {MAX_VIDEO_BYTES // (1024*1024)}MB.",
        )

    # Resolve translation flag
    do_translate = translate_to_english
    if include_translation is not None:
        do_translate = include_translation

    try:
        return MultilingualVideoAnalyzer.analyze_clip(
            video_bytes=video_bytes,
            filename=target_upload.filename or "scene.mp4",
            language_override=language_override,
            translate_to_english=do_translate,
            frame_sampling_interval_sec=frame_sampling_interval_sec,
            segment_interval_sec=segment_interval_sec,
            max_segments=max_segments,
        )
    except Exception as e:
        logger.error(f"Error in movie scene full-clip analysis: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


class YouTubeAnalysisRequest(BaseModel):
    video_url: str = Field(..., description="YouTube video URL or 11-char video ID")
    title: Optional[str] = None
    movie_name: Optional[str] = None
    character: Optional[str] = None
    scene_description: Optional[str] = None
    emotion_tag: Optional[str] = None
    thumbnail_url: Optional[str] = None
    embed_url: Optional[str] = None
    language_override: Optional[str] = None
    translate_to_english: bool = True
    frame_sampling_interval_sec: float = 1.0
    segment_interval_sec: Optional[float] = 3.0
    max_segments: int = 25


@app.get("/api/scenes/video-stream/{video_id}")
def stream_cached_video(video_id: str):
    """Streams a cached YouTube video clip directly to the HTML5 video player."""
    clean_id = YouTubeClipDownloader.clean_video_id(video_id) or video_id
    cached_path = YouTubeClipDownloader.get_cached_video_path(clean_id)
    if not cached_path or not os.path.exists(cached_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cached stream not found for video ID: {clean_id}",
        )
    return FileResponse(cached_path, media_type="video/mp4")


@app.post("/api/scenes/analyze-youtube")
def analyze_youtube_movie_scene(req: YouTubeAnalysisRequest):
    """
    Downloads a targeted YouTube video stream via yt-dlp and executes the complete
    tri-modal deep learning emotion pipeline (Whisper STT, OpenCV Grad-CAM, openSMILE, DistilBERT).
    """
    clean_id = YouTubeClipDownloader.clean_video_id(req.video_url)
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid YouTube URL or Video ID: {req.video_url}",
        )

    # 1. Download or retrieve cached YouTube stream
    try:
        download_info = YouTubeClipDownloader.download_clip(
            url_or_id=clean_id,
            max_duration_sec=300,
        )
    except Exception as e:
        logger.warning(f"yt-dlp download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not download this video, so it was not analyzed: {e}",
        )

    if not download_info or not os.path.exists(download_info["file_path"]):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not download this video, so it was not analyzed.",
        )

    with open(download_info["file_path"], "rb") as vf:
        video_bytes = vf.read()

    result = MultilingualVideoAnalyzer.analyze_clip(
        video_bytes=video_bytes,
        filename=f"youtube_{clean_id}.mp4",
        language_override=req.language_override,
        translate_to_english=req.translate_to_english,
        frame_sampling_interval_sec=req.frame_sampling_interval_sec,
        segment_interval_sec=req.segment_interval_sec,
        max_segments=req.max_segments,
    )
    result["video_stream_url"] = f"/api/scenes/video-stream/{clean_id}"
    result["video_id"] = clean_id
    result["title"] = download_info.get("title")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
