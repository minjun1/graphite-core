"""tests/test_analyzer.py — Unit tests for graphite.pipeline.analyzer (mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphite.pipeline.analyzer import ArgumentAnalyzer, analyze_argument_chain
from graphite.pipeline.verdict import (
    Verdict, VerdictEnum, VerdictRationale,
    ArgumentVerdictEnum,
)


def _make_verdict(claim_text="test claim", verdict=VerdictEnum.SUPPORTED):
    return Verdict(
        claim_id="claim-001",
        claim_text=claim_text,
        verdict=verdict,
        supporting_evidence_ids=[],
        conflicting_evidence_ids=[],
        rationale=VerdictRationale(text="test rationale"),
        needs_human_review=False,
        model_version="test",
        timestamp="2026-01-01T00:00:00Z",
    )


def _mock_analyzer_response(argument_verdicts):
    mock_message = MagicMock()
    mock_message.content = json.dumps({"argument_verdicts": argument_verdicts})
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestArgumentAnalyzer:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_grounded_result(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "The conclusion follows", "verdict": "GROUNDED",
             "rationale_text": "Evidence supports", "contradiction_type": None,
             "needs_human_review": False},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        verdicts = [_make_verdict()]
        results = analyzer.analyze_argument_chain("The memo text", verdicts)

        assert len(results) == 1
        assert results[0].verdict == ArgumentVerdictEnum.GROUNDED

    @patch("graphite.pipeline._client.create_openai_client")
    def test_conclusion_jump(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "Unsupported conclusion", "verdict": "CONCLUSION_JUMP",
             "rationale_text": "Leap in logic", "contradiction_type": None,
             "needs_human_review": True},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        results = analyzer.analyze_argument_chain("memo", [_make_verdict()])

        assert results[0].verdict == ArgumentVerdictEnum.CONCLUSION_JUMP
        assert results[0].needs_human_review is True

    @patch("graphite.pipeline._client.create_openai_client")
    def test_overstated(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "Overstated claim", "verdict": "OVERSTATED",
             "rationale_text": "Exaggerated", "contradiction_type": None,
             "needs_human_review": False},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        results = analyzer.analyze_argument_chain("memo", [_make_verdict()])

        assert results[0].verdict == ArgumentVerdictEnum.OVERSTATED

    @patch("graphite.pipeline._client.create_openai_client")
    def test_empty_verdicts(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([])

        analyzer = ArgumentAnalyzer(api_key="test-key")
        results = analyzer.analyze_argument_chain("memo", [])
        assert results == []


class TestAnalyzeConvenience:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_convenience_function(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "ok", "verdict": "GROUNDED", "rationale_text": "fine",
             "contradiction_type": None, "needs_human_review": False},
        ])

        results = analyze_argument_chain("memo", [_make_verdict()], api_key="test-key")
        assert len(results) == 1


class TestAnalyzerMalformedJSON:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_malformed_json_raises_value_error(self, mock_create):
        """Malformed LLM response raises ValueError, not JSONDecodeError."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "not valid json {{"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        analyzer = ArgumentAnalyzer(api_key="test-key")
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            analyzer.analyze_argument_chain("memo", [_make_verdict()])


class TestCustomPrompt:
    @patch("graphite.pipeline._client.create_openai_client")
    def test_custom_system_prompt_is_used(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_analyzer_response([
            {"text": "ok", "verdict": "GROUNDED", "rationale_text": "fine",
             "contradiction_type": None, "needs_human_review": False},
        ])

        analyzer = ArgumentAnalyzer(api_key="test-key", system_prompt="Custom analyzer prompt.")
        analyzer.analyze_argument_chain("memo", [_make_verdict()])

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "Custom analyzer prompt."
