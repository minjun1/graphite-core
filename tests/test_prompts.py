"""Tests for graphite.pipeline.prompts."""

from graphite.pipeline.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    ANALYZER_SYSTEM_PROMPT,
    PromptSet,
    DEFAULT_PROMPTS,
)


class TestPromptConstants:
    def test_extractor_prompt_is_nonempty_string(self):
        assert isinstance(EXTRACTOR_SYSTEM_PROMPT, str)
        assert len(EXTRACTOR_SYSTEM_PROMPT) > 50

    def test_verifier_prompt_is_nonempty_string(self):
        assert isinstance(VERIFIER_SYSTEM_PROMPT, str)
        assert len(VERIFIER_SYSTEM_PROMPT) > 50

    def test_analyzer_prompt_is_nonempty_string(self):
        assert isinstance(ANALYZER_SYSTEM_PROMPT, str)
        assert len(ANALYZER_SYSTEM_PROMPT) > 50


class TestPromptSet:
    def test_default_prompts_has_all_keys(self):
        assert DEFAULT_PROMPTS.extractor is not None
        assert DEFAULT_PROMPTS.verifier is not None
        assert DEFAULT_PROMPTS.analyzer is not None

    def test_custom_prompt_set(self):
        custom = PromptSet(extractor="Extract medical claims.")
        assert custom.extractor == "Extract medical claims."
        # Others fall back to defaults
        assert custom.verifier == VERIFIER_SYSTEM_PROMPT
        assert custom.analyzer == ANALYZER_SYSTEM_PROMPT

    def test_prompt_set_merge(self):
        base = DEFAULT_PROMPTS
        override = PromptSet(verifier="Medical fact-checker.")
        merged = base.merge(override)
        assert merged.extractor == EXTRACTOR_SYSTEM_PROMPT
        assert merged.verifier == "Medical fact-checker."
        assert merged.analyzer == ANALYZER_SYSTEM_PROMPT


class TestPromptSetMergeSentinel:
    def test_merge_no_args_keeps_base(self):
        """PromptSet() with no overrides should not replace base values."""
        base = PromptSet(extractor="Custom base extractor.")
        no_override = PromptSet()
        merged = base.merge(no_override)
        assert merged.extractor == "Custom base extractor."

    def test_merge_explicit_default_is_override(self):
        """Explicitly passing the default value should still count as an override."""
        base = PromptSet(extractor="Custom base extractor.")
        override = PromptSet(extractor=EXTRACTOR_SYSTEM_PROMPT)
        merged = base.merge(override)
        assert merged.extractor == EXTRACTOR_SYSTEM_PROMPT

    def test_merge_partial_override(self):
        """Override only verifier, keep extractor and analyzer from base."""
        base = PromptSet(extractor="Custom extractor.", analyzer="Custom analyzer.")
        override = PromptSet(verifier="Custom verifier.")
        merged = base.merge(override)
        assert merged.extractor == "Custom extractor."
        assert merged.verifier == "Custom verifier."
        assert merged.analyzer == "Custom analyzer."
