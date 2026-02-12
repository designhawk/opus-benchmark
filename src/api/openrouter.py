"""OpenRouter API client for LLM translations."""

import requests
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter API integration for LLM translation."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/anomalyco/opencode",
            "Title": "OPUS LLM Benchmark",
        }

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str = "arcee-ai/trinity-large-preview:free",
    ) -> str:
        """
        Translate a single text with retry logic (2 retries).

        Args:
            text: Source text to translate
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'de')
            model: OpenRouter model identifier

        Returns:
            Translated text

        Raises:
            RuntimeError: If translation fails after 2 retries
        """
        system_prompt = (
            f"You are a professional translator. "
            f"Translate the following text from {source_lang} to {target_lang}. "
            "Provide only the translation, nothing else."
        )

        for attempt in range(3):  # 1 original + 2 retries
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1024,
                    },
                    timeout=60,
                )

                response.raise_for_status()

                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    raise ValueError("Unexpected response format from OpenRouter")

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/3)")
                if attempt == 2:
                    raise RuntimeError(
                        f"Translation failed after 2 retries: request timeout"
                    )
                time.sleep(2**attempt)

            except requests.exceptions.RequestException as e:
                # Check for rate limiting (429 error)
                response = getattr(e, "response", None)
                if (
                    response is not None
                    and getattr(response, "status_code", None) == 429
                ):
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    logger.warning(
                        f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/3"
                    )
                    time.sleep(wait_time)
                    if attempt == 2:
                        raise RuntimeError(
                            "Translation failed: Rate limit exceeded. Please wait a minute and try again."
                        )
                else:
                    logger.warning(f"Request error: {e} (attempt {attempt + 1}/3)")
                    if attempt == 2:
                        raise RuntimeError(
                            f"Translation failed after 2 retries: {str(e)}"
                        )
                    time.sleep(2**attempt)

            except Exception as e:
                logger.warning(f"Unexpected error: {e} (attempt {attempt + 1}/3)")
                if attempt == 2:
                    raise RuntimeError(f"Translation failed after 2 retries: {str(e)}")
                time.sleep(2**attempt)

        # Should never reach here, but just in case
        raise RuntimeError("Translation failed: Unknown error")

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        model: str = "openrouter",
        max_concurrent: int = 5,
    ) -> List[str]:
        """
        Translate multiple texts.

        Args:
            texts: List of source texts
            source_lang: Source language code
            target_lang: Target language code
            model: OpenRouter model identifier
            max_concurrent: Maximum concurrent requests

        Returns:
            List of translated texts
        """
        results = []

        for i, text in enumerate(texts):
            try:
                translation = self.translate(text, source_lang, target_lang, model)
                results.append(translation)
            except RuntimeError as e:
                logger.error(f"Failed to translate text {i}: {e}")
                results.append(f"[ERROR: {str(e)}]")

        return results

    def check_credits(self) -> Dict:
        """Check remaining OpenRouter credits."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/credits",
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check credits: {e}")
            return {"error": str(e)}
