"""tests/test_cache.py — Unit tests for graphite.cache."""

import pytest
from graphite.cache import PipelineCache


class TestMakeKey:
    def test_deterministic(self):
        k1 = PipelineCache.make_key("src1", "hash1", "v1", "p1", "gemini")
        k2 = PipelineCache.make_key("src1", "hash1", "v1", "p1", "gemini")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = PipelineCache.make_key("src1", "hash1", "v1", "p1", "gemini")
        k2 = PipelineCache.make_key("src2", "hash1", "v1", "p1", "gemini")
        assert k1 != k2

    def test_key_length(self):
        k = PipelineCache.make_key("a", "b", "c", "d", "e")
        assert len(k) == 24


class TestContentHash:
    def test_deterministic(self):
        h1 = PipelineCache.content_hash("hello world")
        h2 = PipelineCache.content_hash("hello world")
        assert h1 == h2

    def test_length(self):
        h = PipelineCache.content_hash("test")
        assert len(h) == 16


class TestCacheOperations:
    def test_put_get_roundtrip(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        cache.put("key1", {"result": "data", "count": 42})
        result = cache.get("key1")
        assert result == {"result": "data", "count": 42}

    def test_has_after_put(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        assert cache.has("key1") is False
        cache.put("key1", {"x": 1})
        assert cache.has("key1") is True

    def test_get_missing_returns_none(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        assert cache.get("nonexistent") is None

    def test_clear(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        cache.put("k1", {"a": 1})
        cache.put("k2", {"b": 2})
        count = cache.clear()
        assert count == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_clear_empty(self, tmp_path):
        cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
        count = cache.clear()
        assert count == 0
