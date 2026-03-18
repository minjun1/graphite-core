"""
graphite/cache.py — Extraction result caching.

5-part cache key prevents pollution when anything changes:
  (source_id, content_hash, extractor_version, prompt_version, model_name)
"""
import hashlib
import json
import os
from typing import Any, Optional


class PipelineCache:
    """File-based cache for pipeline extraction results.

    Stores results as JSON files keyed by a 5-part hash.
    Prevents re-running expensive LLM calls when iterating.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def make_key(
        source_id: str,
        content_hash: str,
        extractor_version: str,
        prompt_version: str,
        model_name: str,
    ) -> str:
        """Build a 5-part cache key."""
        parts = f"{source_id}|{content_hash}|{extractor_version}|{prompt_version}|{model_name}"
        return hashlib.sha256(parts.encode()).hexdigest()[:24]

    @staticmethod
    def content_hash(text: str) -> str:
        """Hash document content for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached result, or None if not cached."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def put(self, key: str, data: Any) -> None:
        """Store result in cache."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def has(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return os.path.exists(os.path.join(self.cache_dir, f"{key}.json"))

    def clear(self) -> int:
        """Clear all cached results. Returns count of deleted files."""
        count = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, f))
                count += 1
        return count
