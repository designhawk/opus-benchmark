"""OPUS API client for language and corpus discovery."""

import requests
from typing import List, Dict, Optional, Set
from pathlib import Path
import json
import logging
import time

logger = logging.getLogger(__name__)


class OpusAPIClient:
    """HTTP client for OPUS API."""

    BASE_URL = "https://opus.nlpl.eu/opusapi/"

    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "opus_benchmark"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OPUS-Benchmark/1.0"})

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached API response."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    cached_at = data.get("cached_at", 0)
                    ttl = data.get("ttl", 3600)
                    if time.time() - cached_at < ttl:
                        return data.get("data")
            except:
                pass
        return None

    def _cache(self, key: str, data: Dict, ttl_seconds: int = 3600):
        """Cache API response."""
        cache_file = self.cache_dir / f"{key}.json"
        cache_data = {
            "cached_at": time.time(),
            "ttl": ttl_seconds,
            "data": data,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

    def list_corpora(self) -> List[str]:
        """Get all available corpora from OPUS API."""
        response = self.session.get(f"{self.BASE_URL}?corpora=True", timeout=30)
        response.raise_for_status()
        return response.json().get("corpora", [])

    def list_languages(
        self, corpus: Optional[str] = None, source: Optional[str] = None
    ) -> List[str]:
        """Get available languages, optionally filtered."""
        params: Dict[str, str] = {}
        if corpus:
            params["corpus"] = corpus
        if source:
            params["source"] = source
        response = self.session.get(
            f"{self.BASE_URL}?languages=True", params=params, timeout=30
        )
        response.raise_for_status()
        return response.json().get("languages", [])

    def list_targets_for_source(
        self, source: str, corpus: Optional[str] = None
    ) -> List[str]:
        """Get available target languages for a source language."""
        params: Dict[str, str] = {"languages": True, "source": source}
        if corpus:
            params["corpus"] = corpus
        response = self.session.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("languages", [])

    def get_corpus_info(self, corpus: str, source: str, target: str) -> Dict:
        """Get corpus metadata including download URL."""
        params = {
            "corpus": corpus,
            "source": source,
            "target": target,
            "version": "latest",
        }
        response = self.session.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_all_corpora_for_pair(
        self, source: str, target: str, max_size_bytes: Optional[int] = None
    ) -> List[Dict]:
        """Fetch all corpora for a language pair with full metadata."""
        cache_key = f"corpora_{source.lower()}_{target.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            corpora = cached
        else:
            url = f"{self.BASE_URL}?source={source.lower()}&target={target.lower()}"
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            corpora = data.get("corpora", [])
            self._cache(cache_key, corpora, ttl_seconds=86400)

        result = []
        for corp in corpora:
            if not corp.get("source") or not corp.get("target"):
                continue

            size_bytes = corp.get("size", 0)
            size_mb = size_bytes / (1024 * 1024) if size_bytes else 0

            corpus_name = corp.get("corpus", "")
            version = corp.get("version", "")
            prep = corp.get("preprocessing", "")

            result.append(
                {
                    "name": corpus_name,
                    "version": version,
                    "preprocessing": prep,
                    "source": corp.get("source", ""),
                    "target": corp.get("target", ""),
                    "alignment_pairs": corp.get("alignment_pairs", 0),
                    "size_bytes": size_bytes,
                    "size_mb": size_mb,
                    "size_formatted": self._format_size(size_mb),
                    "url": corp.get("url", ""),
                    "documents": corp.get("documents", 0),
                }
            )

        if max_size_bytes is not None:
            result = [c for c in result if c["size_bytes"] <= max_size_bytes]

        result.sort(key=lambda x: x["size_bytes"])

        return result

    def _format_size(self, mb: float) -> str:
        """Format size in human readable format."""
        if mb < 1:
            return f"{int(mb * 1024)}KB"
        elif mb < 1000:
            return f"{mb:.0f}MB"
        else:
            return f"{mb / 1024:.1f}GB"

    def search_corpora(self, source: str, target: str, query: str) -> List[Dict]:
        """Search corpora by name substring."""
        all_corpora = self.get_all_corpora_for_pair(source, target)
        query_lower = query.lower()
        return [c for c in all_corpora if query_lower in c["name"].lower()]

    def get_small_corpora(
        self, source: str, target: str, max_mb: float = 1000.0
    ) -> List[Dict]:
        """Get corpora below size threshold (default: 1GB/1000MB)."""
        return self.get_all_corpora_for_pair(
            source, target, max_size_bytes=int(max_mb * 1024 * 1024)
        )

    def get_corpus_by_name(self, source: str, target: str, name: str) -> Optional[Dict]:
        """Get specific corpus by name."""
        all_corpora = self.get_all_corpora_for_pair(source, target)
        for corp in all_corpora:
            if corp["name"].lower() == name.lower():
                return corp
        return None

    def get_corpora_for_pair(self, source: str, target: str) -> List[str]:
        """Get list of unique corpus names for a language pair."""
        corpora = self.get_all_corpora_for_pair(source, target)
        seen = set()
        result = []
        for corp in corpora:
            name = corp["name"]
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def get_download_info(
        self, corpus: str, source: str, target: str
    ) -> Optional[Dict]:
        """Get download URL and metadata for a specific corpus."""
        url = f"{self.BASE_URL}?corpus={corpus}&source={source.lower()}&target={target.lower()}&version=latest"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "corpora" in data and len(data["corpora"]) > 0:
                for corp in data["corpora"]:
                    src = corp.get("source", "").lower()
                    tgt = corp.get("target", "").lower()
                    if src and tgt:
                        if src == source.lower() and tgt == target.lower():
                            prep = corp.get("preprocessing", "")
                            size_bytes = corp.get("size", 0)
                            size_mb = size_bytes / (1024 * 1024) if size_bytes else 0

                            return {
                                "name": corp.get("corpus", ""),
                                "version": corp.get("version", ""),
                                "preprocessing": prep,
                                "source": src,
                                "target": tgt,
                                "alignment_pairs": corp.get("alignment_pairs", 0),
                                "size_bytes": size_bytes,
                                "size_mb": size_mb,
                                "size_formatted": self._format_size(size_mb),
                                "url": corp.get("url", ""),
                                "documents": corp.get("documents", 0),
                            }
        except Exception as e:
            logger.warning(f"Failed to get download info for {corpus}: {e}")
        return None

    def check_pair_available(self, corpus: str, source: str, target: str) -> bool:
        """Check if a language pair is available in a corpus."""
        try:
            targets = self.list_targets_for_source(source, corpus)
            return target.upper() in [t.upper() for t in targets]
        except Exception:
            return False
