"""Multi-language evaluation for OPUS-LLM-Benchmark.

Evaluates translations across multiple language pairs.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.languages import TARGET_LANGUAGES, LANGUAGE_CODES


class MultiLangEvaluator:
    """Evaluate translations across multiple languages."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_language(
        self,
        target_lang: str,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> Dict[str, Any]:
        """Evaluate translations for a single language."""
        from evaluation.metrics import TranslationEvaluator

        evaluator = TranslationEvaluator()

        # references should already be List[List[str]] (one list per sentence)
        # Each inner list contains reference translations for that sentence
        # If references is List[str], wrap each in a list
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        metrics = evaluator.evaluate(references, hypotheses)
        sentence_metrics = evaluator.evaluate_per_sentence(references, hypotheses)

        source_lang = "en"
        lang_info = TARGET_LANGUAGES.get(target_lang, {"name": target_lang})

        result = {
            "source": source_lang,
            "target": target_lang,
            "target_name": lang_info.get("name", target_lang),
            "samples": len(references),
            "metrics": {
                "bleu": round(metrics["bleu"], 2),
                "chrf": round(metrics["chrf"], 2),
            },
            "sentence_metrics": [
                {
                    "bleu": round(s.get("bleu", 0), 2),
                    "chrf": round(s.get("chrf", 0), 2),
                }
                for s in sentence_metrics
            ],
            "evaluated_at": datetime.now().isoformat(),
        }

        self.results[target_lang] = result
        return result

    def evaluate_all_languages(
        self,
        translations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate translations for all languages."""
        all_results = {}

        for target_lang, lang_data in translations.items():
            references = lang_data.get("references", [])
            hypotheses = lang_data.get("hypotheses", [])

            if references and hypotheses:
                result = self.evaluate_language(target_lang, references, hypotheses)
                all_results[target_lang] = result

        return all_results

    def get_ranking(self, metric: str = "bleu") -> List[Tuple[str, float, int]]:
        """Get language ranking by metric."""
        ranking = []
        for lang, result in self.results.items():
            score = result["metrics"].get(metric, 0)
            samples = result.get("samples", 0)
            ranking.append((lang, score, samples))
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all languages."""
        if not self.results:
            return {}

        bleu_scores = [r["metrics"]["bleu"] for r in self.values()]
        chrf_scores = [r["metrics"]["chrf"] for r in self.values()]

        return {
            "num_languages": len(self.results),
            "bleu": {
                "mean": round(statistics.mean(bleu_scores), 2),
                "median": round(statistics.median(bleu_scores), 2),
                "stdev": round(statistics.stdev(bleu_scores), 2)
                if len(bleu_scores) > 1
                else 0,
                "min": round(min(bleu_scores), 2),
                "max": round(max(bleu_scores), 2),
            },
            "chrf": {
                "mean": round(statistics.mean(chrf_scores), 2),
                "median": round(statistics.median(chrf_scores), 2),
                "stdev": round(statistics.stdev(chrf_scores), 2)
                if len(chrf_scores) > 1
                else 0,
                "min": round(min(chrf_scores), 2),
                "max": round(max(chrf_scores), 2),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "results": self.results,
            "summary": self.get_summary_stats(),
            "rankings": {
                "bleu": self.get_ranking("bleu"),
                "chrf": self.get_ranking("chrf"),
            },
        }

    def values(self):
        """Return results values."""
        return self.results.values()


class CheckpointManager:
    """Manage benchmark checkpoints for resume capability."""

    def __init__(self, checkpoint_file: str = "./data/wikimedia/checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default_template()

    def _default_template(self) -> Dict[str, Any]:
        """Get default checkpoint template."""
        return {
            "version": "1.0",
            "status": "in_progress",
            "config": {},
            "languages": {},
            "started_at": None,
            "last_updated": None,
        }

    def save(self):
        """Save checkpoint to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def init_benchmark(
        self,
        config: Dict[str, Any],
        languages: List[str],
    ):
        """Initialize a new benchmark checkpoint."""
        self.data = {
            "version": "1.0",
            "status": "in_progress",
            "config": config,
            "languages": {
                lang: {"status": "pending", "samples": 0} for lang in languages
            },
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        self.save()

    def update_language(
        self,
        lang: str,
        status: str,
        metrics: Optional[Dict[str, float]] = None,
        samples: int = 0,
        result_data: Optional[List[Dict]] = None,
    ):
        """Update status for a language."""
        if lang not in self.data["languages"]:
            self.data["languages"][lang] = {"status": "pending", "samples": 0}

        self.data["languages"][lang].update(
            {
                "status": status,
                "samples": samples,
                "metrics": metrics,
                "completed_at": datetime.now().isoformat()
                if status == "completed"
                else None,
            }
        )

        if result_data:
            self.data["languages"][lang]["results"] = result_data

        self.save()

    def get_pending_languages(self) -> List[str]:
        """Get list of languages not yet completed."""
        return [
            lang
            for lang, data in self.data.get("languages", {}).items()
            if data.get("status") != "completed"
        ]

    def get_completed_languages(self) -> List[str]:
        """Get list of completed languages."""
        return [
            lang
            for lang, data in self.data.get("languages", {}).items()
            if data.get("status") == "completed"
        ]

    def get_language_result(self, lang: str) -> Optional[Dict]:
        """Get results for a specific language."""
        return self.data.get("languages", {}).get(lang)

    def is_benchmark_complete(self) -> bool:
        """Check if all languages are completed."""
        all_langs = set(self.data.get("languages", {}).keys())
        completed = set(self.get_completed_languages())
        return all_langs and completed == all_langs

    def mark_complete(self):
        """Mark benchmark as complete."""
        self.data["status"] = "completed"
        self.data["completed_at"] = datetime.now().isoformat()
        self.save()

    def get_all_results(self) -> Dict[str, Dict]:
        """Get all results from checkpoint."""
        results = {}
        for lang, data in self.data.get("languages", {}).items():
            if data.get("status") == "completed":
                results[lang] = {
                    "references": [],
                    "hypotheses": [],
                    "metrics": data.get("metrics", {}),
                    "samples": data.get("samples", 0),
                }
                if "results" in data:
                    results[lang]["sentences"] = data["results"]
        return results


def merge_checkpoints(checkpoints: List[str], output_file: str):
    """Merge multiple checkpoints into one."""
    merged = {
        "version": "1.0",
        "merged_from": checkpoints,
        "languages": {},
    }

    for cp_file in checkpoints:
        cp_path = Path(cp_file)
        if cp_path.exists():
            with open(cp_path, "r", encoding="utf-8") as f:
                cp_data = json.load(f)
                for lang, data in cp_data.get("languages", {}).items():
                    if data.get("status") == "completed":
                        merged["languages"][lang] = data

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    return merged
