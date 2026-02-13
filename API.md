# Custom Translation API Guide

This guide explains how to replace OpenRouter with other translation APIs like DeepL, Google Translate, Azure, etc.

## Architecture Overview

```
src/
├── api/
│   └── openrouter.py      # OpenRouter API client
├── llm/
│   └── translator.py       # Translation orchestration
├── evaluation/
│   └── metrics.py         # BLEU, chrF++, METEOR scoring
└── cli/
    └── main.py            # CLI commands
```

The `LLMTranslator` class in `src/llm/translator.py` uses `OpenRouterClient` for translations. To use a different API, you need to create a custom client class.

## Required Interface

Any translation API client must implement this interface:

```python
class TranslationClient:
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key

    def translate(
        self,
        text: str,
        source_lang: str,  # ISO code: 'en', 'de', 'fr', etc.
        target_lang: str,  # ISO code: 'de', 'fr', 'ja', etc.
        model: str = None  # Optional model identifier
    ) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Source text to translate
            source_lang: ISO 639-1 language code (e.g., 'en')
            target_lang: ISO 639-1 language code (e.g., 'de')
            model: Optional model/endpoint identifier
            
        Returns:
            Translated text string
            
        Raises:
            RuntimeError: If translation fails
        """
        pass
```

## Adding a New Provider

### Step 1: Create API Client

Create a new file in `src/api/` (e.g., `deepl.py`):

```python
"""DeepL API client for translations."""

import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DeepLClient:
    """DeepL API integration for translation."""

    # DeepL API endpoints
    BASE_URL = "https://api-free.deepl.com/v2"  # Use api.deepl.com for Pro

    # Language code mapping: ISO 639-1 -> DeepL codes
    LANG_MAP = {
        "en": "EN",
        "de": "DE",
        "fr": "FR",
        "es": "ES",
        "it": "IT",
        "nl": "NL",
        "pl": "PL",
        "pt": "PT",
        "ru": "RU",
        "ja": "JA",
        "zh": "ZH",
        "ko": "KO",
        "ar": "AR",
        "cs": "CS",
        "da": "DA",
        "el": "EL",
        "et": "ET",
        "fi": "FI",
        "he": "HE",
        "hu": "HU",
        "id": "ID",
        "lt": "LT",
        "lv": "LV",
        "nb": "NB",
        "ro": "RO",
        "sk": "SK",
        "sl": "SL",
        "sv": "SV",
        "th": "TH",
        "tr": "TR",
        "uk": "UK",
        "vi": "VI",
        "bg": "BG",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _map_lang(self, code: str) -> str:
        """Map ISO code to DeepL code."""
        return self.LANG_MAP.get(code.lower(), code.upper())

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: Optional[str] = None
    ) -> str:
        """
        Translate text using DeepL API.

        Args:
            text: Source text to translate
            source_lang: ISO source language code
            target_lang: ISO target language code
            model: Ignored for DeepL (uses default)

        Returns:
            Translated text

        Raises:
            RuntimeError: If translation fails
        """
        try:
            response = requests.post(
                f"{self.BASE_URL}/translate",
                params={"auth_key": self.api_key},
                data={
                    "text": [text],
                    "source_lang": self._map_lang(source_lang),
                    "target_lang": self._map_lang(target_lang),
                },
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            if "translations" in result and len(result["translations"]) > 0:
                return result["translations"][0]["text"]
            else:
                raise ValueError("Unexpected DeepL response format")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepL translation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Translation error: {str(e)}")

    def check_usage(self) -> dict:
        """Check API usage statistics."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/usage",
                params={"auth_key": self.api_key},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check usage: {e}")
            return {"error": str(e)}
```

### Step 2: Update Translator

Modify `src/llm/translator.py` to accept custom client:

```python
"""LLM translation orchestration."""

from typing import List, Dict, Optional, Type
import time
import logging

from api.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


class LLMTranslator:
    """Orchestrate translations with progress tracking."""

    def __init__(
        self,
        api_key: str,
        model: str = "arcee-ai/trinity-large-preview:free",
        client_class: Optional[Type] = None,
        **client_kwargs
    ):
        """
        Initialize translator.

        Args:
            api_key: API key for translation service
            model: Model identifier (for OpenRouter)
            client_class: Custom client class (optional)
            **client_kwargs: Additional arguments for custom client
        """
        if client_class:
            self.client = client_class(api_key, **client_kwargs)
        else:
            self.client = OpenRouterClient(api_key)
        self.model = model

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_samples: int,
    ) -> Dict[int, str]:
        """Translate a batch of texts."""
        results = {}
        texts_to_translate = texts[:num_samples]

        print(f"Translating {len(texts_to_translate)} texts...")

        for i, text in enumerate(texts_to_translate):
            try:
                translation = self.client.translate(
                    text, source_lang, target_lang, self.model
                )
                results[i] = translation
                print(f"  [{i + 1}/{len(texts_to_translate)}] {translation[:50]}...")

                # Rate limiting
                if i < len(texts_to_translate) - 1:
                    time.sleep(1.0)

            except RuntimeError as e:
                logger.error(f"Translation failed: {e}")
                results[i] = f"[ERROR: {str(e)}]"

        return results
```

### Step 3: Use Custom Client

```python
from src.llm.translator import LLMTranslator
from src.api.deepl import DeepLClient

# Use DeepL instead of OpenRouter
translator = LLMTranslator(
    api_key="YOUR_DEEPL_KEY",
    client_class=DeepLClient
)

results = translator.translate_batch(
    texts=["Hello world"],
    source_lang="en",
    target_lang="de",
    num_samples=1
)
```

## Language Code Mapping

Different APIs use different language codes:

| ISO 639-1 | OpenRouter | DeepL | Google |
|-----------|------------|-------|--------|
| en | en | EN-US | en |
| de | de | DE | de |
| fr | fr | FR | fr |
| zh | zh | ZH | zh-CN |
| ja | ja | JA | ja |

Create a mapping function in your client to handle these differences.

## Testing Your Integration

### Basic Test

```python
from src.api.deepl import DeepLClient

client = DeepLClient("YOUR_API_KEY")
result = client.translate("Hello", "en", "de")
print(result)  # Should print: Hallo
```

### Debug Tips

1. **Check API Key**: Verify your key has sufficient credits
2. **Test Rate Limits**: Add delays between requests
3. **Log Responses**: Print raw API responses during development
4. **Handle Errors**: Wrap API calls in try/except blocks

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Adding Google Translate

```python
"""Google Translate API client."""

import requests
from typing import Optional


class GoogleTranslateClient:
    """Google Cloud Translation API client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://translation.googleapis.com/language/translate/v2"

    def translate(self, text: str, source_lang: str, target_lang: str, model: Optional[str] = None) -> str:
        response = requests.post(
            self.url,
            params={"key": self.api_key},
            data={
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"]["translations"][0]["translatedText"]
```

## Summary

To add a new translation provider:

1. **Create** `src/api/{provider}.py` with client class
2. **Implement** `translate()` method with required signature
3. **Handle** language code mapping internally
4. **Add** error handling and retry logic
5. **Test** with basic translation calls

The benchmark will work with any translation API that returns text!
