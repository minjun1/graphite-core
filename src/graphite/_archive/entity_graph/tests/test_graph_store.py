"""
tests/test_graph_store.py — Tests for the core graph store.
"""

import pytest
from graphite.graph import clamp, generate_quote_hash, GraphStore


class TestQuoteHash:
    def test_deterministic(self):
        h1 = generate_quote_hash("NVIDIA is the sole supplier")
        h2 = generate_quote_hash("NVIDIA is the sole supplier")
        assert h1 == h2

    def test_normalized(self):
        """Extra whitespace should not change the hash."""
        h1 = generate_quote_hash("NVIDIA  is   the sole supplier")
        h2 = generate_quote_hash("NVIDIA is the sole supplier")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = generate_quote_hash("NVIDIA is the sole supplier")
        h2 = generate_quote_hash("nvidia is the sole supplier")
        assert h1 == h2

    def test_empty(self):
        assert generate_quote_hash("") == ""

    def test_length(self):
        h = generate_quote_hash("test")
        assert len(h) == 16


class TestGraphStoreAliases:
    def test_canonicalize(self):
        store = GraphStore.__new__(GraphStore)
        store._aliases = {"GOOG": "GOOGL"}
        store._node_meta = {}
        assert store.canonicalize("GOOG") == "GOOGL"
        assert store.canonicalize("AAPL") == "AAPL"

    def test_canonicalize_case_insensitive(self):
        store = GraphStore.__new__(GraphStore)
        store._aliases = {"GOOG": "GOOGL"}
        store._node_meta = {}
        assert store.canonicalize("goog") == "GOOGL"

    def test_load_aliases(self):
        store = GraphStore.__new__(GraphStore)
        store._aliases = {}
        store._node_meta = {}
        store.load_aliases(
            {"GOOG": "GOOGL", "FB": "META"},
            {"GOOGL": {"name": "Alphabet"}, "META": {"name": "Meta Platforms"}},
        )
        assert store.canonicalize("GOOG") == "GOOGL"
        assert store.canonicalize("FB") == "META"
        assert store.get_node_meta("GOOG") == {"name": "Alphabet"}
