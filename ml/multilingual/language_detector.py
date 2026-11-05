"""
language_detector.py - Automatic Language Identification & Script Analysis
Accurately identifies English (en), Hindi (hi), Malayalam (ml), and Tamil (ta)
using Unicode script block boundaries, character frequency distributions,
and lexical markers with full support for user manual overrides.
"""
import re
import unicodedata
from typing import Dict, Any, Optional

# Supported language definitions
SUPPORTED_LANGUAGES = {
    "en": {"code": "hi", "name": "English", "script": "Latin", "native_name": "English"},
    "hi": {"code": "hi", "name": "Hindi", "script": "Devanagari", "native_name": "हिन्दी"},
    "ml": {"code": "ml", "name": "Malayalam", "script": "Malayalam", "native_name": "മലയാളം"},
    "ta": {"code": "ta", "name": "Tamil", "script": "Tamil", "native_name": "தமிழ்"},
}
SUPPORTED_LANGUAGES["en"]["code"] = "en"

# Unicode script ranges
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "ml": (0x0D00, 0x0D7F),  # Malayalam
}

# Transliterated / Romanized keyword markers for Indic languages in Latin script
ROMANIZED_LEXICON = {
    "hi": {
        "bahut", "achha", "accha", "khush", "gussa", "dukh", "dard", "pyar", "naraaz",
        "kyun", "kaise", "mera", "meri", "hum", "tum", "aap", "nahi", "nahin", "haan",
        "shukriya", "dhanyavaad", "yaar", "bhai", "kya", "hota", "karenge", "mujhe",
    },
    "ta": {
        "vanakkam", "nandri", "romba", "eppadi", "santhosham", "kovam", "kavalai", "bayam",
        "naan", "neenga", "enakku", "illai", "aama", "seri", "thalaiva", "nanba",
        "enna", "yaar", "inge", "ange", "mudiyum", "theriyum", "solunga",
    },
    "ml": {
        "namaskaram", "nanni", "valare", "santhosham", "duhkham", "dhesyam", "pedi",
        "njan", "ningal", "enikku", "illa", "athe", "entha", "engane", "ivide",
        "avidu", "cheyyam", "ariyam", "parayu", "chettan", "chechi", "snehikunnu",
    },
    "en": {
        "the", "and", "is", "in", "it", "you", "that", "he", "was", "for", "on", "are",
        "with", "as", "i", "his", "they", "be", "at", "one", "have", "this", "from",
        "or", "had", "by", "not", "word", "but", "what", "some", "we", "can", "out",
    },
}


class LanguageDetector:
    """
    Robust multilingual language detector for conversational text,
    transcripts, and metadata.
    """

    @classmethod
    def detect(cls, text: str, override_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Detects language from text with optional user manual override.

        Args:
            text: Input string in native Indic script or Latin script.
            override_language: Optional override code ('en', 'hi', 'ml', 'ta').

        Returns:
            Dictionary containing detected language metadata, confidence, and script details.
        """
        raw_text = (text or "").strip()

        # Handle user manual override
        if override_language:
            norm_override = override_language.strip().lower()
            if norm_override in SUPPORTED_LANGUAGES:
                lang_meta = SUPPORTED_LANGUAGES[norm_override]
                # Still compute auto-detection for comparison
                auto_result = cls._detect_raw(raw_text)
                return {
                    "language_code": lang_meta["code"],
                    "language_name": lang_meta["name"],
                    "native_name": lang_meta["native_name"],
                    "script": lang_meta["script"],
                    "confidence": 1.0,
                    "is_override": True,
                    "auto_detected": auto_result,
                    "supported": True,
                }

        auto_result = cls._detect_raw(raw_text)
        return {
            "language_code": auto_result["code"],
            "language_name": auto_result["name"],
            "native_name": auto_result["native_name"],
            "script": auto_result["script"],
            "confidence": round(auto_result["confidence"], 4),
            "is_override": False,
            "script_distribution": auto_result.get("distribution", {}),
            "supported": True,
        }

    @classmethod
    def _detect_raw(cls, text: str) -> Dict[str, Any]:
        """Core internal detection algorithm."""
        if not text:
            return {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "script": "Latin",
                "confidence": 0.5,
                "distribution": {},
            }

        counts = {"hi": 0, "ta": 0, "ml": 0, "latin": 0, "other": 0}
        total_meaningful = 0

        for ch in text:
            code_pt = ord(ch)
            cat = unicodedata.category(ch)
            # Only count letters, marks, and numbers
            if not (cat.startswith("L") or cat.startswith("M")):
                continue

            total_meaningful += 1
            if SCRIPT_RANGES["hi"][0] <= code_pt <= SCRIPT_RANGES["hi"][1]:
                counts["hi"] += 1
            elif SCRIPT_RANGES["ta"][0] <= code_pt <= SCRIPT_RANGES["ta"][1]:
                counts["ta"] += 1
            elif SCRIPT_RANGES["ml"][0] <= code_pt <= SCRIPT_RANGES["ml"][1]:
                counts["ml"] += 1
            elif (0x0041 <= code_pt <= 0x005A) or (0x0061 <= code_pt <= 0x007A):
                counts["latin"] += 1
            else:
                counts["other"] += 1

        if total_meaningful == 0:
            return {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "script": "Latin",
                "confidence": 0.5,
                "distribution": counts,
            }

        # 1. Native Indic Script Predominance
        for lang_code in ["hi", "ta", "ml"]:
            if counts[lang_code] > 0 and (counts[lang_code] / total_meaningful) >= 0.25:
                ratio = counts[lang_code] / total_meaningful
                conf = min(0.99, 0.70 + 0.30 * ratio)
                meta = SUPPORTED_LANGUAGES[lang_code]
                return {
                    "code": lang_code,
                    "name": meta["name"],
                    "native_name": meta["native_name"],
                    "script": meta["script"],
                    "confidence": conf,
                    "distribution": counts,
                }

        # 2. Latin Script: Check if English or Romanized Indic
        tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z]+\b", text)]
        if not tokens:
            return {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "script": "Latin",
                "confidence": 0.6,
                "distribution": counts,
            }

        # Check Romanized lexicon matches
        hits = {"hi": 0, "ta": 0, "ml": 0, "en": 0}
        for tok in tokens:
            for lk in ["hi", "ta", "ml", "en"]:
                if tok in ROMANIZED_LEXICON[lk]:
                    hits[lk] += 1

        best_romanized_indic = max(["hi", "ta", "ml"], key=lambda k: hits[k])
        if hits[best_romanized_indic] > 0 and hits[best_romanized_indic] >= hits["en"]:
            meta = SUPPORTED_LANGUAGES[best_romanized_indic]
            conf = min(0.92, 0.55 + 0.10 * hits[best_romanized_indic])
            return {
                "code": best_romanized_indic,
                "name": f"{meta['name']} (Romanized)",
                "native_name": meta["native_name"],
                "script": "Latin (Transliterated)",
                "confidence": conf,
                "distribution": counts,
            }

        # Default to English
        en_conf = min(0.98, 0.75 + (counts["latin"] / total_meaningful) * 0.23)
        return {
            "code": "en",
            "name": "English",
            "native_name": "English",
            "script": "Latin",
            "confidence": en_conf,
            "distribution": counts,
        }

    @classmethod
    def get_supported_languages(cls) -> Dict[str, Any]:
        """Returns metadata for all supported languages."""
        return {
            "languages": list(SUPPORTED_LANGUAGES.values()),
            "codes": list(SUPPORTED_LANGUAGES.keys()),
            "default": "en",
        }

