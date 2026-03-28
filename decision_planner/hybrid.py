"""Hybrid decision planner."""

from decision_planner.confidence_based import ConfidenceBasedPlanner
from decision_planner.rule_based import RuleBasedPlanner


class HybridPlanner:
    def __init__(self):
        self.rule = RuleBasedPlanner()
        self.conf = ConfidenceBasedPlanner()

    def fit(self, attacks, confidences, severities, targets):
        del severities
        self.rule.fit(attacks, confidences, targets)
        self.conf.fit(confidences, targets)
        return self

    def decide(self, attack, confidence, severity):
        del severity
        if attack == "Normal Traffic":
            return "No Action"

        if confidence > 0.85:
            return "Block"

        return "Alert"
