"""Confidence-based decision planner."""


class ConfidenceBasedPlanner:
    def fit(self, confidences, targets):
        del confidences, targets
        return self

    def decide(self, attack, confidence, severity):
        del severity
        if attack == "Normal Traffic":
            return "No Action"
        if confidence > 0.8:
            return "Block"
        if confidence > 0.6:
            return "Alert"
        return "Monitor"
