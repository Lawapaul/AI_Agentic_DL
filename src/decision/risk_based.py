"""Risk-based decision planner."""


class RiskBasedPlanner:
    def fit(self, confidences, severities, targets):
        del confidences, severities, targets
        return self

    def decide(self, attack, confidence, severity):
        del severity
        if attack == "Normal Traffic":
            return "No Action"

        risk = confidence

        if risk > 0.8:
            return "Block"
        if risk > 0.6:
            return "Alert"
        return "Monitor"
