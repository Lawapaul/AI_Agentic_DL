"""Rule-based decision planner with learnable thresholds."""

from __future__ import annotations

import numpy as np


class RuleBasedPlanner:
    def __init__(self):
        self.normal_action = "No Action"
        self.high_conf_action = "Block"
        self.default_action = "Monitor"
        self.confidence_threshold = 0.8

    def fit(self, attacks, confidences, targets):
        attacks = np.asarray(attacks)
        confidences = np.asarray(confidences, dtype=float)
        targets = np.asarray(targets)

        normal_mask = attacks == "Normal Traffic"
        if normal_mask.any():
            values, counts = np.unique(targets[normal_mask], return_counts=True)
            self.normal_action = values[int(np.argmax(counts))]

        attack_mask = ~normal_mask
        if attack_mask.any():
            best_score = -1.0
            for threshold in np.unique(confidences[attack_mask]):
                high_mask = attack_mask & (confidences >= threshold)
                low_mask = attack_mask & (confidences < threshold)
                high_action = self._majority(targets[high_mask], fallback=self.high_conf_action)
                low_action = self._majority(targets[low_mask], fallback=self.default_action)
                score = np.mean(
                    np.where(
                        normal_mask,
                        self.normal_action,
                        np.where(confidences >= threshold, high_action, low_action),
                    )
                    == targets
                )
                if score > best_score:
                    best_score = score
                    self.confidence_threshold = float(threshold)
                    self.high_conf_action = high_action
                    self.default_action = low_action
        return self

    def decide(self, attack, confidence, severity):
        del severity
        if attack == "Normal Traffic":
            return self.normal_action
        if confidence >= self.confidence_threshold:
            return self.high_conf_action
        return self.default_action

    @staticmethod
    def _majority(values, fallback):
        if len(values) == 0:
            return fallback
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[int(np.argmax(counts))]
