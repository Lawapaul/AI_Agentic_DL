"""Rule-based decision planner."""


class RuleBasedPlanner:
    def decide(self, attack, confidence, severity):
        del severity
        if attack == "Normal Traffic":
            return "No Action"
        if confidence > 0.8:
            return "Block"
        return "Monitor"
