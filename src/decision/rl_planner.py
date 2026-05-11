"""Lightweight Q-learning planner."""

from __future__ import annotations

import random


class RLPlanner:
    def __init__(self):
        self.q_table = {}
        self.actions = ["Block", "Monitor", "Alert", "No Action"]
        self.high_threshold = 0.75
        self.medium_threshold = 0.4

    def fit(self, attacks, confidences, targets, epochs: int = 5, alpha: float = 0.2, epsilon: float = 0.1):
        points = list(zip(attacks, confidences, targets))
        if not points:
            return self
        for _ in range(epochs):
            random.shuffle(points)
            for attack, confidence, target in points:
                action = self.decide(attack, confidence, "unused", epsilon=epsilon)
                reward = 1.0 if action == target else -0.25
                self.update(attack, confidence, action, reward, alpha=alpha)
        return self

    def get_state(self, attack, confidence):
        if confidence > self.high_threshold:
            conf_level = "high"
        elif confidence > self.medium_threshold:
            conf_level = "medium"
        else:
            conf_level = "low"
        return f"{attack}_{conf_level}"

    def decide(self, attack, confidence, severity, epsilon=0.0):
        del severity
        state = self.get_state(attack, confidence)
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if random.random() < epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, attack, confidence, action, reward, alpha=0.1):
        state = self.get_state(attack, confidence)
        if state not in self.q_table:
            self.q_table[state] = {candidate: 0.0 for candidate in self.actions}
        self.q_table[state][action] += alpha * (reward - self.q_table[state][action])
