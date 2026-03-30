"""Independent runner for decision planner phase."""

from __future__ import annotations

from typing import Dict

from decision_planner.probabilistic_planner import ProbabilisticPlanner
from decision_planner.risk_threshold import RiskThresholdPlanner
from decision_planner.rl_agent import RLAgentPlanner
from decision_planner.rule_policy import RulePolicyPlanner


def run_decision_planner(sample_input: Dict[str, object], method: str = "risk_threshold") -> Dict[str, object]:
    """Run standalone decision planner for one sample."""
    planners = {
        "risk_threshold": RiskThresholdPlanner(),
        "rule_policy": RulePolicyPlanner(),
        "probabilistic": ProbabilisticPlanner(),
        "rl_agent": RLAgentPlanner(),
    }

    key = method.strip().lower()
    if key not in planners:
        raise ValueError(f"Unknown planner method: {method}")

    return planners[key].decide(
        risk_score=float(sample_input["risk_score"]),
        attack_type=str(sample_input["attack_type"]),
        llm_explanation=dict(sample_input.get("llm_explanation", {})),
    )
