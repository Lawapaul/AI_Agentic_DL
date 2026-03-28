"""Confidence-based decision planner."""


class ConfidenceBasedPlanner:
    def decide(self, attack, confidence, severity):
        del attack, severity
        if confidence > 0.85:
            return "Block"
        if confidence > 0.6:
            return "Alert + Monitor"
        return "Log Only"
