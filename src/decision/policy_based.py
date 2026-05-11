"""Policy-based decision planner with learned action table."""

from __future__ import annotations

import numpy as np


class PolicyBasedPlanner:
    def __init__(self):
        self.policy = {
            ("Normal Traffic", "Low"): "Monitor",
            ("Normal Traffic", "Medium"): "Monitor",
            ("Normal Traffic", "High"): "Monitor",
            ("Attack", "Low"): "Monitor",
            ("Attack", "Medium"): "Alert",
            ("Attack", "High"): "Block",
        }

    def fit(self, attacks, severities, targets):
        attacks = np.asarray(attacks)
        severities = np.asarray(severities)
        targets = np.asarray(targets)
        for attack in np.unique(attacks):
            for severity in np.unique(severities):
                mask = (attacks == attack) & (severities == severity)
                if mask.any():
                    self.policy[(str(attack), str(severity))] = self._majority(targets[mask], self.policy.get((str(attack), str(severity)), "Monitor"))
        return self

    def decide(self, attack, confidence, severity):
        del confidence
        return self.policy.get((attack, severity), "Monitor")

    @staticmethod
    def _majority(values, fallback):
        if len(values) == 0:
            return fallback
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[int(np.argmax(counts))]
