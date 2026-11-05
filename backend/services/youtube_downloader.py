"""
youtube_downloader.py - High-Performance YouTube Stream Downloader using yt-dlp
Safely downloads targeted clip streams for multimodal emotion recognition analysis
with duration capping, format selection, progress callbacks, and local video caching.
"""
import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import yt_dlp

logger = logging.getLogger(__name__)

# Cache directory for downloaded clips placed OUTSIDE backend to prevent uvicorn hot-reload loops
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cached_clips"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_YOUTUBE_DURATION_SEC = 300  # Cap at 5 minutes for fast processing


class YouTubeClipDownloader:
    """
    Downloads YouTube video clips for offline deep learning emotion analysis
    using yt-dlp with speed optimization and format constraints.
    """

    @classmethod
    def clean_video_id(cls, url_or_id: str) -> Optional[str]:
        """Extracts standard 11-char YouTube video ID."""
        if not url_or_id:
            return None
        url_or_id = url_or_id.strip()
        patterns = [
            r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/watch\?v=|\&v=)([A-Za-z0-9_-]{11})",
            r"^([A-Za-z0-9_-]{11})$",
        ]
        for p in patterns:
            m = re.search(p, url_or_id)
            if m:
                return m.group(1)
        return None

    @classmethod
    def get_cached_video_path(cls, video_id: str) -> Optional[str]:
        """Returns path to cached video file if already downloaded."""
        potential = CACHE_DIR / f"{video_id}.mp4"
        if potential.exists() and potential.stat().st_size > 10000:
            return str(potential)
        return None

    @classmethod
    def download_clip(
        cls,
        url_or_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_duration_sec: int = MAX_YOUTUBE_DURATION_SEC,
    ) -> Dict[str, Any]:
        """
        Downloads a YouTube clip to an MP4 container (optimized 360p/480p).
        Returns metadata, cached path, and duration.
        """
        video_id = cls.clean_video_id(url_or_id)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or Video ID: {url_or_id}")

        # Check existing cache first
        cached_file = cls.get_cached_video_path(video_id)
        if cached_file:
            logger.info(f"Using cached video clip for ID {video_id}: {cached_file}")
            if progress_callback:
                progress_callback(100.0, "Loaded from cache")
            return {
                "video_id": video_id,
                "file_path": cached_file,
                "is_cached": True,
            }

        target_file = str(CACHE_DIR / f"{video_id}.mp4")
        watch_url = f"https://www.youtube.com/watch?v={video_id}"

        def _yt_hook(d):
            if d.get("status") == "downloading" and progress_callback:
                total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
                downloaded = d.get("downloaded_bytes", 0)
                pct = min(95.0, (downloaded / total_bytes) * 100.0)
                progress_callback(round(pct, 1), f"Downloading stream: {pct:.0f}%")
            elif d.get("status") == "finished" and progress_callback:
                progress_callback(100.0, "Stream download complete")

        # Optimal format for AI processing: fast download, 360p resolution, using android/web client to avoid PO-Token 403
        ydl_opts = {
            "format": "ba/b[height<=360]/best[height<=360]/best",
            "outtmpl": target_file,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "socket_timeout": 8,
            "retries": 1,
            "extractor_retries": 1,
            "nocheckcertificate": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"]
                }
            },
            "progress_hooks": [_yt_hook],
        }

        logger.info(f"Downloading YouTube video {watch_url} via yt-dlp to {target_file}...")
        if progress_callback:
            progress_callback(10.0, "Initiating YouTube stream connection...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(watch_url, download=True)
            title = info.get("title", video_id)
            duration = info.get("duration", 0)

        if not os.path.exists(target_file) or os.path.getsize(target_file) < 1000:
            raise RuntimeError(f"Failed to download YouTube video: {video_id}")

        return {
            "video_id": video_id,
            "title": title,
            "duration": duration,
            "file_path": target_file,
            "is_cached": False,
        }

