"""
test_scene_discovery.py - Automated Test Suite for SerpApi Movie Scene Discovery & Full-Clip Analysis
Tests:
  - Query formatting across movie, description, emotion, character, and language
  - Video ID extraction from various YouTube URL formats
  - Multi-language search filtering (English, Hindi, Malayalam, Tamil)
  - Resilient mock fallback when API key is unset or quota exhausted
  - Featured scene catalogue categorization
  - Full-clip movie scene video analysis pipeline with configurable intervals
"""
import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.serp_scene_service import (
    SerpMovieSceneService,
    extract_youtube_video_id,
    MOCK_SCENE_CATALOGUE,
)
from backend.services.video_multilingual_analyzer import MultilingualVideoAnalyzer
from backend.main import (
    search_movie_scenes,
    get_featured_scenes,
)


def test_serp_query_builder():
    """Verify YouTube search query formatting combines movie, character, emotion, and language."""
    q1 = SerpMovieSceneService.build_search_query(
        movie_name="Interstellar",
        scene_description="watching video messages",
        language="en",
        emotion="sadness",
        character_name="Cooper",
    )
    assert "Interstellar" in q1
    assert "Cooper" in q1
    assert "sadness scene" in q1

    q_hi = SerpMovieSceneService.build_search_query(
        movie_name="3 Idiots",
        scene_description="hospital awakening",
        language="hi",
        emotion="joy",
        character_name="Rancho",
    )
    assert "3 Idiots" in q_hi
    assert "Hindi" in q_hi
    assert "joy scene" in q_hi

    q_ml = SerpMovieSceneService.build_search_query(
        movie_name="Kumbalangi Nights",
        scene_description="breakdown",
        language="ml",
        emotion="sadness",
    )
    assert "Kumbalangi Nights" in q_ml
    assert "Malayalam" in q_ml


def test_extract_youtube_video_id():
    """Verify video ID extraction across diverse URL structures."""
    cases = [
        ("https://www.youtube.com/watch?v=v77IOwXg3B4", "v77IOwXg3B4"),
        ("https://youtu.be/r9LdOtwQd5c", "r9LdOtwQd5c"),
        ("https://www.youtube.com/embed/qM-gZintWDc", "qM-gZintWDc"),
        ("S1XpD0r4m38", "S1XpD0r4m38"),
    ]
    for url, expected_id in cases:
        assert extract_youtube_video_id(url) == expected_id


def test_search_scenes_with_mock_fallback():
    """Verify search returns standardized scene results with all required fields."""
    res = SerpMovieSceneService.search_scenes(
        movie_name="The Dark Knight",
        emotion="anger",
        language="en",
    )
    assert res["status"] == "success"
    assert "results" in res
    assert len(res["results"]) >= 1

    first = res["results"][0]
    required_keys = ["video_id", "title", "thumbnail_url", "channel_name", "duration", "video_url", "embed_url"]
    for k in required_keys:
        assert k in first, f"Key '{k}' missing from scene result"
        assert first[k], f"Key '{k}' must not be empty"


def test_search_scenes_multilingual_hindi():
    """Verify Hindi movie scene search filters to Hindi results."""
    res = SerpMovieSceneService.search_scenes(
        movie_name="Taare Zameen Par",
        language="hi",
        emotion="sadness",
    )
    assert res["status"] == "success"
    assert len(res["results"]) >= 1
    assert any(s["language"] == "hi" for s in res["results"])


def test_search_scenes_multilingual_malayalam():
    """Verify Malayalam movie scene search returns Malayalam scenes."""
    res = SerpMovieSceneService.search_scenes(
        movie_name="Kumbalangi Nights",
        language="ml",
    )
    assert res["status"] == "success"
    assert len(res["results"]) >= 1
    assert any(s["language"] == "ml" for s in res["results"])


def test_search_scenes_multilingual_tamil():
    """Verify Tamil movie scene search returns Tamil scenes."""
    res = SerpMovieSceneService.search_scenes(
        movie_name="Vikram Vedha",
        language="ta",
    )
    assert res["status"] == "success"
    assert len(res["results"]) >= 1
    assert any(s["language"] == "ta" for s in res["results"])


def test_featured_scenes_endpoint():
    """Verify featured movie scenes catalogue contains entries for en, hi, ml, and ta."""
    res = get_featured_scenes()
    assert res["status"] == "success"
    assert "featured" in res
    for lang in ["en", "hi", "ml", "ta"]:
        assert lang in res["featured"]
        assert len(res["featured"][lang]) >= 3


def test_search_scenes_endpoint_handler():
    """Verify GET /api/scenes/search endpoint handler functions cleanly."""
    res = search_movie_scenes(movie="Interstellar", language="en", emotion="sadness")
    assert res["status"] == "success"
    assert "results" in res
    assert len(res["results"]) >= 1


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


def test_full_clip_movie_scene_analysis():
    """Verify full-clip movie scene video analysis with configurable frame and segment intervals."""
    res = MultilingualVideoAnalyzer.analyze_clip(
        video_bytes=_blank_video_bytes(),
        filename="scene_clip.mp4",
        language_override="en",
        frame_sampling_interval_sec=0.5,
        segment_interval_sec=2.0,
        max_segments=5,
    )
    assert res["status"] == "success"
    assert "scene_summary" in res
    assert "timeline" in res
    assert len(res["timeline"]) >= 1

    first = res["timeline"][0]
    assert "timestamp" in first
    assert "fused_emotion" in first
    assert "modality_predictions" in first
    assert "text" in first["modality_predictions"]
    assert "acoustic" in first["modality_predictions"]
    assert "visual" in first["modality_predictions"]
    assert "visual_gradcam_image" in first

    # No speech, voice or face means no emotion is claimed
    assert first["fused_emotion"] == "unknown"
    assert res["scene_summary"]["dominant_emotion"] == "unknown"


def test_unreadable_video_is_rejected():
    """Verify a file with no video or audio gives an error, not a report."""
    with pytest.raises(ValueError):
        MultilingualVideoAnalyzer.analyze_clip(
            video_bytes=b"\x00\x00\x00 ftypmp42" + b"\x00" * 3500,
            filename="broken.mp4",
        )



def test_youtube_clip_downloader_url_cleaning():
    """Verify video ID extraction from diverse YouTube URL patterns."""
    from backend.services.youtube_downloader import YouTubeClipDownloader

    assert YouTubeClipDownloader.clean_video_id("https://www.youtube.com/watch?v=yM3kO4_88Q8") == "yM3kO4_88Q8"
    assert YouTubeClipDownloader.clean_video_id("https://youtu.be/S1XpD0r4m38?t=10") == "S1XpD0r4m38"
    assert YouTubeClipDownloader.clean_video_id("https://www.youtube.com/embed/v77IOwXg3B4") == "v77IOwXg3B4"
    assert YouTubeClipDownloader.clean_video_id("https://www.youtube.com/shorts/8TLwVhmV-HE?feature=share") == "8TLwVhmV-HE"
    assert YouTubeClipDownloader.clean_video_id("r9LdOtwQd5c") == "r9LdOtwQd5c"
    assert YouTubeClipDownloader.clean_video_id("invalid-url-string") is None


def test_failed_youtube_download_is_not_analyzed(monkeypatch):
    """Verify a failed download returns an error instead of an invented analysis."""
    from fastapi import HTTPException
    from backend.main import analyze_youtube_movie_scene, YouTubeAnalysisRequest
    from backend.services.youtube_downloader import YouTubeClipDownloader

    def fail_download(*args, **kwargs):
        raise RuntimeError("This video is unavailable")

    monkeypatch.setattr(YouTubeClipDownloader, "download_clip", fail_download)
    request = YouTubeAnalysisRequest(video_url="https://www.youtube.com/watch?v=yM3kO4_88Q8")
    with pytest.raises(HTTPException) as error:
        analyze_youtube_movie_scene(request)
    assert error.value.status_code == 502


def test_youtube_endpoint_analyzes_downloaded_file(monkeypatch, tmp_path):
    """Verify the YouTube endpoint runs the real pipeline on the downloaded file."""
    from backend.main import analyze_youtube_movie_scene, YouTubeAnalysisRequest
    from backend.services.youtube_downloader import YouTubeClipDownloader

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(_blank_video_bytes())
    monkeypatch.setattr(YouTubeClipDownloader, "download_clip", lambda **kwargs: {"file_path": str(clip)})

    res = analyze_youtube_movie_scene(YouTubeAnalysisRequest(video_url="yM3kO4_88Q8", max_segments=4))
    assert res["video_id"] == "yM3kO4_88Q8"
    assert res["is_live_inference"] is True
    assert len(res["timeline"]) >= 1
