"""Decision planner package with lazy-loaded strategy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "RuleBasedPlanner",
    "ConfidenceBasedPlanner",
    "RiskBasedPlanner",
    "HybridPlanner",
    "RLPlanner",
    "LLMPlanner",
    "PolicyBasedPlanner",
    "run_decision_planner",
]


def __getattr__(name: str):
    mapping = {
        "RuleBasedPlanner": ("src.decision.rule_based", "RuleBasedPlanner"),
        "ConfidenceBasedPlanner": ("src.decision.confidence_based", "ConfidenceBasedPlanner"),
        "RiskBasedPlanner": ("src.decision.risk_based", "RiskBasedPlanner"),
        "HybridPlanner": ("src.decision.hybrid", "HybridPlanner"),
        "RLPlanner": ("src.decision.rl_planner", "RLPlanner"),
        "LLMPlanner": ("src.decision.llm_planner", "LLMPlanner"),
        "PolicyBasedPlanner": ("src.decision.policy_based", "PolicyBasedPlanner"),
        "run_decision_planner": ("src.decision.pipelines", "run_decision_planner"),
    }
    if name not in mapping:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = mapping[name]
    return getattr(import_module(module_name), attr_name)
