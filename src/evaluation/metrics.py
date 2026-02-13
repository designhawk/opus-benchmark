"""Translation evaluation metrics (BLEU, chrF, METEOR)."""

import sacrebleu
from typing import List, Dict, Tuple
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
import nltk
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """Compute translation evaluation metrics."""

    def __init__(self):
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                logger.warning("Could not download NLTK punkt")
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                logger.warning("Could not download NLTK punkt_tab")
        try:
            nltk.data.find("wordnet")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
            except Exception:
                logger.warning("Could not download NLTK wordnet")

    def evaluate(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            references: List of reference translations (can have multiple per source)
            hypotheses: List of hypothesis translations

        Returns:
            Dictionary with BLEU, chrF, and METEOR scores
        """
        return {
            "bleu": self.bleu(references, hypotheses),
            "chrf": self.chrf(references, hypotheses),
            "meteor": self.meteor(references, hypotheses),
        }

    def evaluate_per_sentence(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> List[Dict[str, float]]:
        """
        Compute metrics per sentence.

        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations

        Returns:
            List of per-sentence metrics
        """
        results = []

        for hyp, refs in zip(hypotheses, references):
            sentence_metrics = {
                "bleu": self._sentence_bleu(refs, hyp),
                "chrf": self._sentence_chrf(refs, hyp),
                "meteor": self._sentence_meteor(refs, hyp),
            }
            results.append(sentence_metrics)

        return results

    def bleu(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """
        Compute corpus-level BLEU score (0-100 scale).

        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations

        Returns:
            BLEU score (0-100)
        """
        try:
            bleu = sacrebleu.corpus_bleu(hypotheses, references)
            return bleu.score
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0

    def chrf(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """
        Compute corpus-level chrF++ score (0-100 scale).

        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations

        Returns:
            chrF++ score (0-100)
        """
        try:
            chrf = sacrebleu.corpus_chrf(hypotheses, references)
            return chrf.score
        except Exception as e:
            logger.error(f"chrF calculation failed: {e}")
            return 0.0

    def meteor(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """
        Compute corpus-level METEOR score (0-100 scale).

        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations

        Returns:
            METEOR score (0-100)
        """
        try:
            scores = []
            for hyp, refs in zip(hypotheses, references):
                hyp_tokens = hyp.split()
                ref_tokens = [r.split() for r in refs]
                score = nltk_meteor_score(ref_tokens, hyp_tokens)
                scores.append(score)
            return sum(scores) / len(scores) * 100 if scores else 0.0
        except Exception as e:
            logger.error(f"METEOR calculation failed: {e}")
            return 0.0

    def _sentence_bleu(
        self,
        references: List[str],
        hypothesis: str,
    ) -> float:
        """Compute sentence-level BLEU."""
        try:
            bleu = sacrebleu.sentence_bleu(hypothesis, references)
            return bleu.score
        except Exception:
            return 0.0

    def _sentence_chrf(
        self,
        references: List[str],
        hypothesis: str,
    ) -> float:
        """Compute sentence-level chrF."""
        try:
            chrf = sacrebleu.sentence_chrf(hypothesis, references)
            return chrf.score
        except Exception:
            return 0.0

    def _sentence_meteor(
        self,
        references: List[str],
        hypothesis: str,
    ) -> float:
        """Compute sentence-level METEOR."""
        try:
            # Tokenize references and hypothesis
            ref_tokens = [r.split() for r in references]
            hyp_tokens = hypothesis.split()
            score = nltk_meteor_score(ref_tokens, hyp_tokens)
            return score * 100
        except Exception as e:
            logger.error(f"METEOR sentence failed: {e}")
            return 0.0
