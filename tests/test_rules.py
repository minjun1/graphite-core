"""tests/test_rules.py — Unit tests for graphite.rules."""

import pytest
from graphite.rules import RuleResult, ScoreBreakdown, BaseRuleEngine


class TestRuleResult:
    def test_fields(self):
        rr = RuleResult(
            rule_id="R01", rule_name="Sole Source",
            triggered=True, weight_delta=0.15,
            explanation="Single supplier", category="supply_risk",
        )
        assert rr.rule_id == "R01"
        assert rr.triggered is True
        assert rr.weight_delta == 0.15
        assert rr.category == "supply_risk"

    def test_default_category(self):
        rr = RuleResult(
            rule_id="R01", rule_name="X",
            triggered=False, weight_delta=0.0, explanation="N/A",
        )
        assert rr.category == ""


class TestScoreBreakdown:
    def _make_breakdown(self):
        rules = [
            RuleResult(rule_id="R01", rule_name="Sole Source", triggered=True, weight_delta=0.15, explanation="yes", category="risk"),
            RuleResult(rule_id="R02", rule_name="Diversified", triggered=False, weight_delta=0.0, explanation="no", category="risk"),
            RuleResult(rule_id="R03", rule_name="Revenue Conc", triggered=True, weight_delta=0.12, explanation="yes", category="revenue"),
        ]
        return ScoreBreakdown(
            base_score=0.5,
            rule_results=rules,
            final_score=0.77,
            raw_delta=0.27,
            applied_delta=0.27,
            confidence="HIGH",
            verdict="SUPPORTED",
            verdict_reason="Strong evidence",
        )

    def test_triggered_rules(self):
        bd = self._make_breakdown()
        triggered = bd.triggered_rules
        assert len(triggered) == 2
        assert all(r.triggered for r in triggered)

    def test_total_delta_returns_applied_delta(self):
        bd = self._make_breakdown()
        assert bd.total_delta == bd.applied_delta
        assert bd.total_delta == 0.27

    def test_to_dict(self):
        bd = self._make_breakdown()
        d = bd.to_dict()
        assert d["base_score"] == 0.5
        assert d["final_score"] == 0.77
        assert d["verdict"] == "SUPPORTED"
        assert d["confidence"] == "HIGH"
        assert d["policy_version"] == "v1"
        assert len(d["triggered_rules"]) == 2
        assert d["triggered_rules"][0]["rule_id"] == "R01"

    def test_to_dict_rounds_values(self):
        bd = ScoreBreakdown(
            base_score=0.123456789,
            final_score=0.987654321,
            raw_delta=0.111111,
            applied_delta=0.111111,
        )
        d = bd.to_dict()
        assert d["base_score"] == 0.1235
        assert d["final_score"] == 0.9877


class TestBaseRuleEngine:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseRuleEngine()

    def test_concrete_subclass(self):
        class MyEngine(BaseRuleEngine):
            def compute_score(self, edge_data, counter_signals=None):
                return ScoreBreakdown(base_score=1.0, final_score=1.0)

        engine = MyEngine()
        result = engine.compute_score({"edge_type": "PRODUCES"})
        assert isinstance(result, ScoreBreakdown)
        assert result.final_score == 1.0
