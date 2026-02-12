"""Multi-language configuration for OPUS-LLM-Benchmark.

Contains language codes, names, and metadata for 32 target languages.
"""

from typing import Dict, List, Optional

TARGET_LANGUAGES: Dict[str, Dict[str, str]] = {
    "ar": {
        "name": "Arabic",
        "native": "العربية",
        "family": "Afro-Asiatic",
        "script": "Arabic",
    },
    "bg": {
        "name": "Bulgarian",
        "native": "Български",
        "family": "Indo-European",
        "script": "Cyrillic",
    },
    "cs": {
        "name": "Czech",
        "native": "Čeština",
        "family": "Indo-European",
        "script": "Latin",
    },
    "da": {
        "name": "Danish",
        "native": "Dansk",
        "family": "Indo-European",
        "script": "Latin",
    },
    "de": {
        "name": "German",
        "native": "Deutsch",
        "family": "Indo-European",
        "script": "Latin",
    },
    "el": {
        "name": "Greek",
        "native": "Ελληνικά",
        "family": "Indo-European",
        "script": "Greek",
    },
    "en": {
        "name": "English",
        "native": "English",
        "family": "Indo-European",
        "script": "Latin",
    },
    "es": {
        "name": "Spanish",
        "native": "Español",
        "family": "Indo-European",
        "script": "Latin",
    },
    "et": {
        "name": "Estonian",
        "native": "Eesti",
        "family": "Uralic",
        "script": "Latin",
    },
    "fi": {"name": "Finnish", "native": "Suomi", "family": "Uralic", "script": "Latin"},
    "fr": {
        "name": "French",
        "native": "Français",
        "family": "Indo-European",
        "script": "Latin",
    },
    "he": {
        "name": "Hebrew",
        "native": "עברית",
        "family": "Afro-Asiatic",
        "script": "Hebrew",
    },
    "hu": {
        "name": "Hungarian",
        "native": "Magyar",
        "family": "Uralic",
        "script": "Latin",
    },
    "id": {
        "name": "Indonesian",
        "native": "Bahasa Indonesia",
        "family": "Austronesian",
        "script": "Latin",
    },
    "it": {
        "name": "Italian",
        "native": "Italiano",
        "family": "Indo-European",
        "script": "Latin",
    },
    "ja": {
        "name": "Japanese",
        "native": "日本語",
        "family": "Japonic",
        "script": "Japanese",
    },
    "ko": {
        "name": "Korean",
        "native": "한국어",
        "family": "Koreanic",
        "script": "Hangul",
    },
    "lt": {
        "name": "Lithuanian",
        "native": "Lietuvių",
        "family": "Indo-European",
        "script": "Latin",
    },
    "lv": {
        "name": "Latvian",
        "native": "Latviešu",
        "family": "Indo-European",
        "script": "Latin",
    },
    "nb": {
        "name": "Norwegian",
        "native": "Norsk Bokmål",
        "family": "Indo-European",
        "script": "Latin",
    },
    "nl": {
        "name": "Dutch",
        "native": "Nederlands",
        "family": "Indo-European",
        "script": "Latin",
    },
    "pl": {
        "name": "Polish",
        "native": "Polski",
        "family": "Indo-European",
        "script": "Latin",
    },
    "pt": {
        "name": "Portuguese",
        "native": "Português",
        "family": "Indo-European",
        "script": "Latin",
    },
    "ro": {
        "name": "Romanian",
        "native": "Română",
        "family": "Indo-European",
        "script": "Latin",
    },
    "ru": {
        "name": "Russian",
        "native": "Русский",
        "family": "Indo-European",
        "script": "Cyrillic",
    },
    "sk": {
        "name": "Slovak",
        "native": "Slovenčina",
        "family": "Indo-European",
        "script": "Latin",
    },
    "sl": {
        "name": "Slovenian",
        "native": "Slovenščina",
        "family": "Indo-European",
        "script": "Latin",
    },
    "sv": {
        "name": "Swedish",
        "native": "Svenska",
        "family": "Indo-European",
        "script": "Latin",
    },
    "th": {"name": "Thai", "native": "ไทย", "family": "Tai-Kadai", "script": "Thai"},
    "tr": {
        "name": "Turkish",
        "native": "Türkçe",
        "family": "Turkic",
        "script": "Latin",
    },
    "uk": {
        "name": "Ukrainian",
        "native": "Українська",
        "family": "Indo-European",
        "script": "Cyrillic",
    },
    "vi": {
        "name": "Vietnamese",
        "native": "Tiếng Việt",
        "family": "Austroasiatic",
        "script": "Latin",
    },
    "zh": {
        "name": "Chinese",
        "native": "中文",
        "family": "Sino-Tibetan",
        "script": "Chinese",
    },
}

LANGUAGE_CODES: List[str] = sorted([k for k in TARGET_LANGUAGES.keys() if k != "en"])

LANGUAGE_FAMILIES: Dict[str, List[str]] = {
    "Indo-European": [
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "he",
        "hu",
        "it",
        "lt",
        "lv",
        "nb",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    ],
    "Uralic": ["et", "fi", "hu"],
    "Japonic": ["ja"],
    "Koreanic": ["ko"],
    "Sino-Tibetan": ["zh"],
    "Turkic": ["tr"],
    "Afro-Asiatic": ["ar", "he"],
    "Austronesian": ["id", "vi"],
    "Tai-Kadai": ["th"],
}

SCRIPT_GROUPS: Dict[str, List[str]] = {
    "Latin": [
        "cs",
        "da",
        "de",
        "es",
        "et",
        "fi",
        "fr",
        "hu",
        "id",
        "it",
        "lt",
        "lv",
        "nb",
        "nl",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
        "tr",
        "vi",
    ],
    "Cyrillic": ["bg", "ru", "uk"],
    "Greek": ["el"],
    "Hebrew": ["he"],
    "Japanese": ["ja"],
    "Chinese": ["zh"],
    "Thai": ["th"],
}

DEFAULT_SAMPLES = 10

DEFAULT_LANGUAGES: List[str] = [
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "he",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "nb",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "th",
    "tr",
    "uk",
    "vi",
    "zh",
]


def get_language_info(code: str) -> Optional[Dict[str, str]]:
    """Get information for a language code."""
    return TARGET_LANGUAGES.get(code.lower())


def validate_language_codes(codes: List[str]) -> List[str]:
    """Validate and return a list of valid language codes."""
    valid = []
    for code in codes:
        code = code.strip().lower()
        if code in TARGET_LANGUAGES:
            valid.append(code)
        elif code in [v["name"].lower() for v in TARGET_LANGUAGES.values()]:
            for k, v in TARGET_LANGUAGES.items():
                if v["name"].lower() == code.lower():
                    valid.append(k)
                    break
    return list(dict.fromkeys(valid))


def get_default_corpus_for_languages(languages: List[str]) -> str:
    """Get the best corpus for a list of languages."""
    return "wikimedia"


def format_language_list(codes: List[str], max_display: int = 5) -> str:
    """Format a list of language codes for display."""
    if len(codes) <= max_display:
        return ", ".join(f"{c.upper()}" for c in codes)
    else:
        shown = ", ".join(f"{c.upper()}" for c in codes[:max_display])
        return f"{shown} ... (+{len(codes) - max_display} more)"
