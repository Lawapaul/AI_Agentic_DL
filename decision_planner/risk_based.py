"""Risk-based decision planner."""


class RiskBasedPlanner:
    def decide(self, attack, confidence, severity):
        del attack
        severity_map = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
        risk = 0.6 * confidence + 0.4 * severity_map.get(severity, 0.5)

        if risk > 0.8:
            return "Block"
        if risk > 0.5:
            return "Alert"
        return "Monitor"
