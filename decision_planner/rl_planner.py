"""Lightweight Q-table based planner."""


class RLPlanner:
    def __init__(self):
        self.q_table = {}
        self.actions = ["Block", "Monitor", "Alert"]

    def get_state(self, attack, confidence):
        level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        return f"{attack}_{level}"

    def decide(self, attack, confidence, severity):
        del severity
        state = self.get_state(attack, confidence)
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        return max(self.q_table[state], key=self.q_table[state].get)
