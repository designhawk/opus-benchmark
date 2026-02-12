"""LLM translation orchestration."""

from typing import List, Dict
import time
import logging

from api.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


class LLMTranslator:
    """Orchestrate LLM translations with progress tracking."""

    def __init__(self, api_key: str, model: str = "arcee-ai/trinity-large-preview:free"):
        self.client = OpenRouterClient(api_key)
        self.model = model

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_samples: int,
    ) -> Dict[int, str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts
            source_lang: Source language code
            target_lang: Target language code
            num_samples: Number of samples to process

        Returns:
            Dictionary mapping index to translation
        """
        results: Dict[int, str] = {}
        texts_to_translate = texts[:num_samples]

        print(f"Translating {len(texts_to_translate)} texts...")

        for i, text in enumerate(texts_to_translate):
            try:
                translation = self.client.translate(
                    text, source_lang, target_lang, self.model
                )
                results[i] = translation
                print(
                    f"  [{i + 1}/{len(texts_to_translate)}] Translated: {translation[:50]}..."
                )

                # Rate limiting: add delay between requests to avoid 429 errors
                # Free tier has stricter limits, so we add a 1-second delay
                if i < len(texts_to_translate) - 1:  # Don't delay after the last one
                    time.sleep(1.0)

            except RuntimeError as e:
                logger.error(f"Translation failed for text {i}: {e}")
                results[i] = f"[ERROR: {str(e)}]"

        return results

    def translate_single(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate a single text.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        return self.client.translate(text, source_lang, target_lang, self.model)

    def check_api_status(self) -> Dict:
        """Check OpenRouter API status."""
        return self.client.check_credits()
