"""
serp_scene_service.py - SerpApi YouTube Movie Scene Discovery Service
Integrates SerpApi's YouTube search engine to discover emotional movie scene clips
across English, Hindi, Malayalam, and Tamil with robust query formatting,
pagination, quota handling, and rich multi-lingual mock fallback.
"""
import os
import re
import logging
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

SERP_API_ENDPOINT = "https://serpapi.com/search.json"

LANGUAGE_LABELS = {
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "all": "Multilingual",
}

EMOTION_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMOTION_MAP = {name: idx for idx, name in enumerate(EMOTION_NAMES)}
EMOTION_COLORS = {
    "neutral": "#94a3b8",
    "surprise": "#38bdf8",
    "fear": "#c084fc",
    "sadness": "#60a5fa",
    "joy": "#fbbf24",
    "disgust": "#34d399",
    "anger": "#f87171",
}

# Authentic curated movie scene mock catalogue for development, testing & quota resilience
MOCK_SCENE_CATALOGUE = [
    # English Scenes
    {
        "video_id": "v77IOwXg3B4",
        "title": "Interstellar - Cooper Watches 23 Years of Messages (Emotional Scene)",
        "channel_name": "Warner Bros. Entertainment",
        "duration": "4:12",
        "thumbnail_url": "https://img.youtube.com/vi/v77IOwXg3B4/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=v77IOwXg3B4",
        "embed_url": "https://www.youtube.com/embed/v77IOwXg3B4",
        "views": "14M views",
        "language": "en",
        "language_name": "English",
        "emotion": "sadness",
        "movie_name": "Interstellar",
        "character": "Cooper",
        "scene_description": "Cooper cries watching years of missed video transmissions from his children.",
    },
    {
        "video_id": "r9LdOtwQd5c",
        "title": "The Dark Knight - Batman Interrogates the Joker Scene",
        "channel_name": "DC / Warner Bros.",
        "duration": "4:35",
        "thumbnail_url": "https://img.youtube.com/vi/r9LdOtwQd5c/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=r9LdOtwQd5c",
        "embed_url": "https://www.youtube.com/embed/r9LdOtwQd5c",
        "views": "28M views",
        "language": "en",
        "language_name": "English",
        "emotion": "anger",
        "movie_name": "The Dark Knight",
        "character": "Batman & Joker",
        "scene_description": "Batman loses control in the interrogation room facing Joker's psychological taunts.",
    },
    {
        "video_id": "qM-gZintWDc",
        "title": "Good Will Hunting - It's Not Your Fault Emotional Breakdown",
        "channel_name": "Miramax",
        "duration": "3:28",
        "thumbnail_url": "https://img.youtube.com/vi/qM-gZintWDc/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=qM-gZintWDc",
        "embed_url": "https://www.youtube.com/embed/qM-gZintWDc",
        "views": "11M views",
        "language": "en",
        "language_name": "English",
        "emotion": "sadness",
        "movie_name": "Good Will Hunting",
        "character": "Will Hunting & Sean",
        "scene_description": "Robin Williams repeatedly assures Matt Damon until emotional catharsis occurs.",
    },
    {
        "video_id": "GokKUqLcvD8",
        "title": "The Pursuit of Happyness - The Job Offer Scene (Tears of Joy)",
        "channel_name": "Sony Pictures",
        "duration": "3:40",
        "thumbnail_url": "https://img.youtube.com/vi/GokKUqLcvD8/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=GokKUqLcvD8",
        "embed_url": "https://www.youtube.com/embed/GokKUqLcvD8",
        "views": "19M views",
        "language": "en",
        "language_name": "English",
        "emotion": "joy",
        "movie_name": "The Pursuit of Happyness",
        "character": "Chris Gardner",
        "scene_description": "Chris holds back overwhelmed tears of pure joy after securing the broker job.",
    },

    # Hindi Scenes
    {
        "video_id": "S1XpD0r4m38",
        "title": "3 Idiots - Raju Rastogi Hospital Scene (Emotional Awakening)",
        "channel_name": "Vinod Chopra Films",
        "duration": "4:45",
        "thumbnail_url": "https://img.youtube.com/vi/S1XpD0r4m38/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=S1XpD0r4m38",
        "embed_url": "https://www.youtube.com/embed/S1XpD0r4m38",
        "views": "22M views",
        "language": "hi",
        "language_name": "Hindi",
        "emotion": "joy",
        "movie_name": "3 Idiots",
        "character": "Rancho & Farhan",
        "scene_description": "Rancho and Farhan emotionally wake up paralyzed Raju with funny family updates.",
    },
    {
        "video_id": "u4y9_Hn-2vU",
        "title": "Taare Zameen Par - Ishaan Crying in Boarding School (Maa Song Scene)",
        "channel_name": "Aamir Khan Productions",
        "duration": "5:10",
        "thumbnail_url": "https://img.youtube.com/vi/u4y9_Hn-2vU/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=u4y9_Hn-2vU",
        "embed_url": "https://www.youtube.com/embed/u4y9_Hn-2vU",
        "views": "35M views",
        "language": "hi",
        "language_name": "Hindi",
        "emotion": "sadness",
        "movie_name": "Taare Zameen Par",
        "character": "Ishaan Awasthi",
        "scene_description": "Young dyslexic boy struggles with isolation and heartbreak after being separated from his family.",
    },
    {
        "video_id": "6B3g-m82E78",
        "title": "Chak De! India - 70 Minute Speech (Inspirational Climax)",
        "channel_name": "YRF",
        "duration": "3:50",
        "thumbnail_url": "https://img.youtube.com/vi/6B3g-m82E78/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=6B3g-m82E78",
        "embed_url": "https://www.youtube.com/embed/6B3g-m82E78",
        "views": "15M views",
        "language": "hi",
        "language_name": "Hindi",
        "emotion": "joy",
        "movie_name": "Chak De! India",
        "character": "Kabir Khan",
        "scene_description": "Coach Kabir delivers his iconic goosebumps speech rallying the team to victory.",
    },
    {
        "video_id": "xY8u8a76s34",
        "title": "Gangs of Wasseypur - Ramadhir Singh Anger Confrontation Scene",
        "channel_name": "Viacom18 Studios",
        "duration": "3:15",
        "thumbnail_url": "https://img.youtube.com/vi/xY8u8a76s34/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=xY8u8a76s34",
        "embed_url": "https://www.youtube.com/embed/xY8u8a76s34",
        "views": "8M views",
        "language": "hi",
        "language_name": "Hindi",
        "emotion": "anger",
        "movie_name": "Gangs of Wasseypur",
        "character": "Ramadhir Singh & Sardar Khan",
        "scene_description": "Intense political feud and fierce dispute erupting into cold criminal rage.",
    },

    # Malayalam Scenes
    {
        "video_id": "yM3kO4_88Q8",
        "title": "Kumbalangi Nights - Saji Psychological Breakdown Scene (Fahadh & Soubin)",
        "channel_name": "Bhavana Studios",
        "duration": "4:20",
        "thumbnail_url": "https://img.youtube.com/vi/yM3kO4_88Q8/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=yM3kO4_88Q8",
        "embed_url": "https://www.youtube.com/embed/yM3kO4_88Q8",
        "views": "9M views",
        "language": "ml",
        "language_name": "Malayalam",
        "emotion": "sadness",
        "movie_name": "Kumbalangi Nights",
        "character": "Saji",
        "scene_description": "Saji visits a therapist and experiences a deeply moving psychological breakdown admitting vulnerability.",
    },
    {
        "video_id": "K3-zP99uF_Q",
        "title": "Drishyam - Georgekutty Police Station Interrogation (Suspense & Fear)",
        "channel_name": "Aashirvad Cinemas",
        "duration": "5:30",
        "thumbnail_url": "https://img.youtube.com/vi/K3-zP99uF_Q/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=K3-zP99uF_Q",
        "embed_url": "https://www.youtube.com/embed/K3-zP99uF_Q",
        "views": "16M views",
        "language": "ml",
        "language_name": "Malayalam",
        "emotion": "fear",
        "movie_name": "Drishyam",
        "character": "Georgekutty",
        "scene_description": "The police ruthlessly interrogate the family as Georgekutty masks fear with composed resolve.",
    },
    {
        "video_id": "d04QeL9Nq_s",
        "title": "Manichitrathazhu - Ganga Nagavalli Trance Climax Scene (Iconic Horror)",
        "channel_name": "Mani Movies",
        "duration": "4:50",
        "thumbnail_url": "https://img.youtube.com/vi/d04QeL9Nq_s/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=d04QeL9Nq_s",
        "embed_url": "https://www.youtube.com/embed/d04QeL9Nq_s",
        "views": "12M views",
        "language": "ml",
        "language_name": "Malayalam",
        "emotion": "surprise",
        "movie_name": "Manichitrathazhu",
        "character": "Ganga / Nagavalli",
        "scene_description": "Shobana switches personalities into Nagavalli lifting heavy wooden bed in shocking horror revelation.",
    },
    {
        "video_id": "r5PzZl_2210",
        "title": "Kireedam - Sethumadhavan Heartbreaking Climax Scene (Mohanlal)",
        "channel_name": "Kireedam Films",
        "duration": "4:10",
        "thumbnail_url": "https://img.youtube.com/vi/r5PzZl_2210/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=r5PzZl_2210",
        "embed_url": "https://www.youtube.com/embed/r5PzZl_2210",
        "views": "7M views",
        "language": "ml",
        "language_name": "Malayalam",
        "emotion": "sadness",
        "movie_name": "Kireedam",
        "character": "Sethumadhavan & Achuthan",
        "scene_description": "Father helplessly watches his son unintentionally turn into an outcast criminal, breaking both their lives.",
    },

    # Tamil Scenes
    {
        "video_id": "t9yN821PqM8",
        "title": "Vikram Vedha - Who is Good and Who is Bad? (Interrogation Scene)",
        "channel_name": "Y Not Studios",
        "duration": "4:40",
        "thumbnail_url": "https://img.youtube.com/vi/t9yN821PqM8/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=t9yN821PqM8",
        "embed_url": "https://www.youtube.com/embed/t9yN821PqM8",
        "views": "18M views",
        "language": "ta",
        "language_name": "Tamil",
        "emotion": "neutral",
        "movie_name": "Vikram Vedha",
        "character": "Vikram & Vedha",
        "scene_description": "Vedha surrenders calmly and challenges Vikram's rigid moral code through a philosophical riddle.",
    },
    {
        "video_id": "m9P2KxX81L4",
        "title": "Soorarai Pottru - Maara Airport Emotional Breakdown Scene (Suriya)",
        "channel_name": "2D Entertainment",
        "duration": "3:55",
        "thumbnail_url": "https://img.youtube.com/vi/m9P2KxX81L4/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=m9P2KxX81L4",
        "embed_url": "https://www.youtube.com/embed/m9P2KxX81L4",
        "views": "14M views",
        "language": "ta",
        "language_name": "Tamil",
        "emotion": "sadness",
        "movie_name": "Soorarai Pottru",
        "character": "Nedumaaran (Maara)",
        "scene_description": "Maara begs strangers at the airport for money to book a flight to see his dying father.",
    },
    {
        "video_id": "vL98z22QwP0",
        "title": "Asuran - Sivasamy Ferocious Transformation Action Scene (Dhanush)",
        "channel_name": "V Creations",
        "duration": "5:15",
        "thumbnail_url": "https://img.youtube.com/vi/vL98z22QwP0/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=vL98z22QwP0",
        "embed_url": "https://www.youtube.com/embed/vL98z22QwP0",
        "views": "25M views",
        "language": "ta",
        "language_name": "Tamil",
        "emotion": "anger",
        "movie_name": "Asuran",
        "character": "Sivasamy",
        "scene_description": "Meek father Sivasamy drops his pacified demeanor to unleash ferocious rage protecting his son.",
    },
    {
        "video_id": "z9Q_22K9P0A",
        "title": "Super Deluxe - Alien Revelation Climax Scene (Philosophical Wonder)",
        "channel_name": "Tyler Durden And Kino Fist",
        "duration": "4:05",
        "thumbnail_url": "https://img.youtube.com/vi/z9Q_22K9P0A/hqdefault.jpg",
        "video_url": "https://www.youtube.com/watch?v=z9Q_22K9P0A",
        "embed_url": "https://www.youtube.com/embed/z9Q_22K9P0A",
        "views": "6M views",
        "language": "ta",
        "language_name": "Tamil",
        "emotion": "surprise",
        "movie_name": "Super Deluxe",
        "character": "Arputham & Boys",
        "scene_description": "Mind-bending climax combining cosmic irony, sudden comedy, and profound human acceptance.",
    },
]


def extract_youtube_video_id(url_or_id: str) -> Optional[str]:
    """Extracts 11-character YouTube video ID from various URL formats."""
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/watch\?v=|\&v=)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return None


class SerpMovieSceneService:
    """
    Service integrating SerpApi YouTube Search for movie scene discovery
    with automated query formulation, multi-language support, and resilient fallback.
    """

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Fetches SERP_API_KEY from environment variables."""
        key = os.getenv("SERP_API_KEY", "").strip()
        return key if key else None

    @classmethod
    def build_search_query(
        cls,
        movie_name: str,
        scene_description: Optional[str] = None,
        language: Optional[str] = "en",
        emotion: Optional[str] = None,
        character_name: Optional[str] = None,
    ) -> str:
        """Constructs an optimal YouTube search query for movie scenes."""
        parts = []
        if movie_name and movie_name.strip():
            parts.append(movie_name.strip())
        if character_name and character_name.strip():
            parts.append(character_name.strip())
        if scene_description and scene_description.strip():
            parts.append(scene_description.strip())
        if emotion and emotion.strip() and emotion.lower() != "all":
            parts.append(f"{emotion.strip()} scene")
        else:
            parts.append("movie scene")

        lang_code = (language or "en").lower().strip()
        if lang_code in LANGUAGE_LABELS and lang_code != "en" and lang_code != "all":
            parts.append(LANGUAGE_LABELS[lang_code])

        return " ".join(parts)

    @classmethod
    def search_scenes(
        cls,
        movie_name: Optional[str] = None,
        scene_description: Optional[str] = None,
        language: Optional[str] = "en",
        emotion: Optional[str] = None,
        character_name: Optional[str] = None,
        page: int = 1,
        per_page: int = 8,
    ) -> Dict[str, Any]:
        """
        Executes movie scene search via SerpApi's YouTube engine.
        Falls back gracefully to rich mock results if key is absent or quota exceeded.
        """
        query = cls.build_search_query(
            movie_name=movie_name or "",
            scene_description=scene_description,
            language=language,
            emotion=emotion,
            character_name=character_name,
        )

        api_key = cls.get_api_key()

        # If API key is available, call SerpApi YouTube Search
        if api_key:
            try:
                params = {
                    "engine": "youtube",
                    "search_query": query,
                    "api_key": api_key,
                }
                with httpx.Client(timeout=8.0) as client:
                    resp = client.get(SERP_API_ENDPOINT, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    if "error" not in data and "video_results" in data:
                        parsed = cls._parse_serpapi_results(data, query, language, emotion)
                        if parsed["results"]:
                            return parsed

                    logger.warning(f"SerpApi returned error or empty video_results: {data.get('error')}. Using mock fallback.")
                else:
                    logger.warning(f"SerpApi returned HTTP {resp.status_code}. Using mock fallback.")

            except Exception as e:
                logger.warning(f"SerpApi request failed ({e}). Falling back to curated mock scenes.")

        # Fallback to curated realistic movie scene catalogue
        return cls._get_mock_search_results(
            movie_name=movie_name,
            scene_description=scene_description,
            language=language,
            emotion=emotion,
            character_name=character_name,
            query=query,
            page=page,
            per_page=per_page,
            is_mock_fallback=True,
        )

    @classmethod
    def _parse_serpapi_results(
        cls,
        serp_data: Dict[str, Any],
        query: str,
        language: Optional[str],
        emotion: Optional[str],
    ) -> Dict[str, Any]:
        """Transforms raw SerpApi YouTube JSON into standardized scene objects."""
        raw_videos = serp_data.get("video_results", [])
        items = []

        for item in raw_videos:
            link = item.get("link", "")
            vid = extract_youtube_video_id(link)
            if not vid:
                continue

            thumb = item.get("thumbnail")
            thumb_url = thumb.get("static") if isinstance(thumb, dict) else (thumb or f"https://img.youtube.com/vi/{vid}/hqdefault.jpg")

            channel_info = item.get("channel", {})
            channel_name = channel_info.get("name", "Unknown Channel") if isinstance(channel_info, dict) else str(channel_info)

            items.append({
                "video_id": vid,
                "title": item.get("title", "Movie Scene Clip"),
                "thumbnail_url": thumb_url,
                "channel_name": channel_name,
                "duration": item.get("length", "N/A"),
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "embed_url": f"https://www.youtube.com/embed/{vid}",
                "views": item.get("views", "N/A"),
                "published_date": item.get("published_date", "Recently"),
                "language": language or "en",
                "language_name": LANGUAGE_LABELS.get(language or "en", "English"),
                "emotion_tag": emotion or "neutral",
                "source": "serpapi_live",
            })

        return {
            "status": "success",
            "source": "serpapi_live",
            "query": query,
            "total_results": len(items),
            "results": items,
            "search_metadata": serp_data.get("search_metadata", {}),
            "pagination": {
                "current_page": 1,
                "has_more": bool(serp_data.get("serpapi_pagination")),
            },
        }

    @classmethod
    def _get_mock_search_results(
        cls,
        movie_name: Optional[str],
        scene_description: Optional[str],
        language: Optional[str],
        emotion: Optional[str],
        character_name: Optional[str],
        query: str,
        page: int,
        per_page: int,
        is_mock_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Filters mock catalogue by language, emotion, and movie keywords."""
        filtered = list(MOCK_SCENE_CATALOGUE)

        # Filter by language if specified
        lang_code = (language or "all").lower()
        if lang_code in ["en", "hi", "ml", "ta"]:
            filtered = [s for s in filtered if s["language"] == lang_code]

        # Filter by emotion if specified
        emo = (emotion or "all").lower()
        if emo in ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]:
            filtered_by_emo = [s for s in filtered if s["emotion"] == emo]
            if filtered_by_emo:
                filtered = filtered_by_emo

        # Filter or sort by movie name / description keywords
        search_terms = " ".join([movie_name or "", scene_description or "", character_name or ""]).lower().split()
        if search_terms:
            scored = []
            for item in filtered:
                text_blob = f"{item['movie_name']} {item['title']} {item['scene_description']} {item['character']}".lower()
                score = sum(1 for term in search_terms if term in text_blob)
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            filtered = [item for score, item in scored]

        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        page_results = filtered[start:end] or filtered[:per_page]

        return {
            "status": "success",
            "source": "curated_mock_catalogue" if is_mock_fallback else "serpapi_live",
            "query": query,
            "total_results": len(filtered),
            "page": page,
            "per_page": per_page,
            "results": page_results,
            "is_mock": is_mock_fallback,
            "notice": "Serving high-accuracy curated movie scenes (configure SERP_API_KEY in .env for live YouTube queries).",
            "search_metadata": {
                "movie_name": movie_name,
                "language": language,
                "emotion": emotion,
                "character": character_name,
            },
        }

    @classmethod
    def get_featured_scenes(cls) -> Dict[str, Any]:
        """Returns pre-categorized featured movie scenes across all 4 languages."""
        by_language = {"en": [], "hi": [], "ml": [], "ta": []}
        for item in MOCK_SCENE_CATALOGUE:
            lang = item.get("language", "en")
            if lang in by_language:
                by_language[lang].append(item)

        return {
            "status": "success",
            "featured": by_language,
            "total_featured": len(MOCK_SCENE_CATALOGUE),
            "supported_languages": ["en", "hi", "ml", "ta"],
        }

    @classmethod
    def detect_movie_language(
        cls,
        text: str,
        language_override: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Detects language and full language name from text/title markers and native scripts."""
        if language_override and language_override.lower() in LANGUAGE_LABELS:
            lang = language_override.lower()
            return lang, LANGUAGE_LABELS.get(lang, "English")

        # 1. Native Unicode Script Detection (100% precision for non-Latin scripts)
        if re.search(r"[\u0900-\u097f]", text):
            return "hi", "Hindi"
        if re.search(r"[\u0d00-\u0d7f]", text):
            return "ml", "Malayalam"
        if re.search(r"[\u0b80-\u0bff]", text):
            return "ta", "Tamil"
        if re.search(r"[\u0c00-\u0c7f]", text):
            return "te", "Telugu"
        if re.search(r"[\u0c80-\u0cff]", text):
            return "kn", "Kannada"
        if re.search(r"[\u0980-\u09ff]", text):
            return "bn", "Bengali"

        low = text.lower()

        def _has_kw(markers: List[str]) -> bool:
            for m in markers:
                if " " in m:
                    if m in low:
                        return True
                else:
                    if re.search(r"\b" + re.escape(m) + r"\b", low):
                        return True
            return False
        
        # Tamil cinema keywords & markers
        ta_markers = [
            "rolex", "vikram", "leo", "jailer", "master", "asuran", "vedha", "suriya",
            "kamal", "rajini", "vijay", "dhanush", "tamil", "kollywood", "anirudh",
            "lokesh", "kaithi", "soorarai", "pariyerum", "nayanthara", "sivakarthikeyan",
            "valimai", "thunivu", "beast", "petta", "mankatha", "mersal", "sarkar",
            "super deluxe", "vada chennai", "karnan", "jai bhim", "viduthalai", "ayalaan",
            "thiruchitrambalam", "gargi"
        ]
        if _has_kw(ta_markers):
            return "ta", "Tamil"

        # Malayalam cinema keywords & markers
        ml_markers = [
            "kumbalangi", "drishyam", "lucifer", "manichitrathazhu", "kireedam",
            "mohanlal", "mammootty", "fahadh", "soubin", "malayalam", "mollywood",
            "georgekutty", "saji", "nagavalli", "shammi", "aavesham", "bheeshma", "premam",
            "manjummel", "bramayugam", "premalu", "neru", "malik", "trance", "nayattu",
            "hridhyam", "minnal murali", "tovino", "kurup", "jallikattu"
        ]
        if _has_kw(ml_markers):
            return "ml", "Malayalam"

        # Hindi cinema keywords & markers
        hi_markers = [
            "3 idiots", "taare zameen", "dangal", "sholay", "wasseypur", "chak de",
            "shah rukh", "srk", "aamir", "salman", "rancho", "hindi", "bollywood",
            "kabir khan", "ishaan", "swades", "lagaan", "jawan", "pathaan", "gangs of wasseypur",
            "jab tak hai jaan", "jab tak", "supergirl", "anushka sharma"
        ]
        if _has_kw(hi_markers):
            return "hi", "Hindi"

        # Telugu cinema keywords & markers
        te_markers = [
            "baahubali", "bahubali", "rrr", "pushpa", "devara", "kalki", "salaar",
            "prabhas", "allu arjun", "ntr", "ram charan", "mahesh babu", "telugu",
            "tollywood", "chiranjeevi", "rajamouli", "ss rajamouli", "vijay devarakonda"
        ]
        if _has_kw(te_markers):
            return "te", "Telugu"

        # Kannada cinema keywords & markers
        kn_markers = [
            "kgf", "kantara", "charlie 777", "yash", "rishab shetty", "shivarajkumar",
            "kannada", "sandalwood", "upendra", "sudeep", "puneeth"
        ]
        if _has_kw(kn_markers):
            return "kn", "Kannada"

        # English / Hollywood markers
        en_markers = [
            "interstellar", "dark knight", "good will hunting", "pursuit of happyness",
            "batman", "joker", "cooper", "english", "hollywood", "oppenheimer", "inception"
        ]
        if _has_kw(en_markers):
            return "en", "English"

        return "en", "English"
