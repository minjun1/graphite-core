"""tests/test_alphaearth.py — Unit tests for graphite.adapters.alphaearth."""

import pytest
import numpy as np
from graphite.adapters.alphaearth import AlphaEarthAdapter, EMBEDDING_DIM


class TestCacheRoundtrip:
    def test_write_and_read(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        adapter._write_cache("test_node", 2017, emb)
        result = adapter._read_cache("test_node", 2017)

        assert result is not None
        np.testing.assert_array_almost_equal(result, emb)

    def test_read_missing_returns_none(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        assert adapter._read_cache("missing", 2017) is None


class TestGetEmbedding:
    def test_returns_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("PORT_HOUSTON", 2017, emb)

        result = adapter.get_embedding(29.7, -95.2, year=2017, node_id="PORT_HOUSTON")
        np.testing.assert_array_almost_equal(result, emb)

    def test_raises_when_not_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        with pytest.raises(FileNotFoundError, match="No AlphaEarth embedding"):
            adapter.get_embedding(0.0, 0.0, year=2017)


class TestGetEmbeddingSafe:
    def test_returns_none_when_not_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        result = adapter.get_embedding_safe(0.0, 0.0, year=2017)
        assert result is None

    def test_returns_embedding_when_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("test", 2017, emb)
        result = adapter.get_embedding_safe(0.0, 0.0, year=2017, node_id="test")
        assert result is not None


class TestGetAreaEmbedding:
    def test_returns_cached_bbox_embedding(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        bbox = (29.0, -96.0, 30.0, -95.0)
        cache_key = f"bbox_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
        adapter._write_cache(cache_key, 2017, emb)

        result = adapter.get_area_embedding(bbox, year=2017)
        np.testing.assert_array_almost_equal(result, emb)

    def test_falls_back_to_get_embedding(self, tmp_path):
        """When bbox not cached, get_area_embedding calls get_embedding with bbox key."""
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        bbox = (29.0, -96.0, 30.0, -95.0)
        with pytest.raises(FileNotFoundError):
            adapter.get_area_embedding(bbox, year=2017)


class TestListCachedAndStats:
    def test_list_cached(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("node_a", 2017, emb)
        adapter._write_cache("node_b", 2017, emb)

        cached = adapter.list_cached(2017)
        assert set(cached) == {"node_a", "node_b"}

    def test_list_cached_empty_year(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        assert adapter.list_cached(2020) == []

    def test_cache_stats(self, tmp_path):
        adapter = AlphaEarthAdapter(cache_dir=str(tmp_path / "cache"))
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        adapter._write_cache("a", 2017, emb)
        adapter._write_cache("b", 2017, emb)
        adapter._write_cache("c", 2018, emb)

        stats = adapter.cache_stats()
        assert stats["2017"] == 2
        assert stats["2018"] == 1
