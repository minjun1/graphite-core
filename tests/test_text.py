"""tests/test_text.py — Unit tests for graphite.text."""

import pytest
from graphite.text import (
    sha1_hex, sha256_hex, normalize_text, clip_quote,
    split_into_paragraphs, score_paragraph,
    find_best_paragraph_for_quote,
    register_strategy, build_context, _strategies,
)


class TestHashFunctions:
    def test_sha1_deterministic(self):
        assert sha1_hex("hello") == sha1_hex("hello")
        assert len(sha1_hex("hello")) == 40

    def test_sha256_deterministic(self):
        assert sha256_hex("hello") == sha256_hex("hello")
        assert len(sha256_hex("hello")) == 64

    def test_sha1_differs_from_sha256(self):
        assert sha1_hex("hello") != sha256_hex("hello")


class TestNormalizeText:
    def test_collapses_newlines(self):
        assert normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_collapses_spaces(self):
        assert normalize_text("a   b") == "a b"

    def test_strips(self):
        assert normalize_text("  hello  ") == "hello"


class TestClipQuote:
    def test_short_unchanged(self):
        assert clip_quote("short text") == "short text"

    def test_truncates_with_ellipsis(self):
        result = clip_quote("a" * 300, max_chars=10)
        assert len(result) == 11
        assert result.endswith("\u2026")

    def test_strips_input(self):
        assert clip_quote("  hello  ") == "hello"


class TestSplitIntoParagraphs:
    def test_splits_on_double_newline(self):
        text = ("a" * 100) + "\n\n" + ("b" * 100)
        result = split_into_paragraphs(text)
        assert len(result) == 2

    def test_min_len_filters(self):
        text = "short\n\n" + ("long" * 30)
        result = split_into_paragraphs(text, min_len=80)
        assert len(result) == 1

    def test_max_paras_limits(self):
        text = "\n\n".join(["x" * 100 for _ in range(10)])
        result = split_into_paragraphs(text, max_paras=3)
        assert len(result) == 3

    def test_empty_text(self):
        assert split_into_paragraphs("") == []


class TestScoreParagraph:
    def test_counts_keywords(self):
        assert score_paragraph("cobalt mining in Congo", ["cobalt", "congo"]) == 2

    def test_case_insensitive(self):
        assert score_paragraph("COBALT Mining", ["cobalt"]) == 1

    def test_no_match(self):
        assert score_paragraph("hello world", ["cobalt"]) == 0


class TestFindBestParagraphForQuote:
    def test_exact_substring_match(self):
        paras = ["First paragraph about nothing.", "Apple is a key supplier of components."]
        idx, hash_val = find_best_paragraph_for_quote(paras, "Apple is a key supplier")
        assert idx == 1
        assert len(hash_val) == 12

    def test_word_overlap_fallback(self):
        paras = ["Cobalt mining operations in Congo.", "Lithium reserves in Chile."]
        idx, _ = find_best_paragraph_for_quote(paras, "Congo cobalt production")
        assert idx == 0

    def test_empty_quote(self):
        idx, hash_val = find_best_paragraph_for_quote(["para"], "")
        assert idx == -1
        assert hash_val == ""

    def test_empty_paragraphs(self):
        idx, hash_val = find_best_paragraph_for_quote([], "some quote")
        assert idx == -1


class TestStrategyRegistry:
    def test_builtin_strategies_registered(self):
        assert "default" in _strategies
        assert "usgs_country_mineral" in _strategies
        assert "sec_minerals" in _strategies
        assert "sec_generic" in _strategies

    def test_register_custom_strategy(self):
        def my_strategy(paragraphs, **kwargs):
            return "custom"
        register_strategy("_test_custom", my_strategy)
        assert "_test_custom" in _strategies

    def test_build_context_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown context strategy"):
            build_context(None, strategy="nonexistent_strategy_xyz")

    def test_build_context_dispatches(self):
        class FakeDoc:
            paragraphs = ["para one " * 20, "para two " * 20]

        result = build_context(FakeDoc(), strategy="default")
        assert isinstance(result, str)
        assert len(result) > 0
