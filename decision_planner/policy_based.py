"""Policy-based decision planner."""


class PolicyBasedPlanner:
    def decide(self, attack, confidence, severity):
        del confidence
        if attack != "Normal Traffic" and severity == "High":
            return "Block"
        if attack != "Normal Traffic" and severity == "Medium":
            return "Alert"
        return "Monitor"
