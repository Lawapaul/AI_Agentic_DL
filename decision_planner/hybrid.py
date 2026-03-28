"""Hybrid decision planner."""

from decision_planner.confidence_based import ConfidenceBasedPlanner
from decision_planner.rule_based import RuleBasedPlanner


class HybridPlanner:
    def __init__(self):
        self.rule = RuleBasedPlanner()
        self.conf = ConfidenceBasedPlanner()

    def decide(self, attack, confidence, severity):
        baseline = self.rule.decide(attack, confidence, severity)
        if baseline == "No Action":
            return baseline
        if confidence > 0.85:
            return "Block"
        return self.conf.decide(attack, confidence, severity)
