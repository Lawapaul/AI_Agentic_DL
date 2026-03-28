"""Confidence-based decision planner with learned thresholds."""

from __future__ import annotations

import numpy as np


class ConfidenceBasedPlanner:
    def __init__(self):
        self.high_threshold = 0.85
        self.medium_threshold = 0.6
        self.high_action = "Block"
        self.medium_action = "Alert + Monitor"
        self.low_action = "Log Only"

    def fit(self, confidences, targets):
        confidences = np.asarray(confidences, dtype=float)
        targets = np.asarray(targets)
        grid = np.unique(confidences)
        best_score = -1.0

        for medium in grid:
            for high in grid:
                if high < medium:
                    continue
                predictions = np.array(
                    [
                        self.high_action if c >= high else self.medium_action if c >= medium else self.low_action
                        for c in confidences
                    ],
                    dtype=object,
                )
                predictions = self._refit_actions(predictions, confidences, targets, medium, high)
                score = np.mean(predictions == targets)
                if score > best_score:
                    best_score = score
                    self.medium_threshold = float(medium)
                    self.high_threshold = float(high)
                    self.high_action = self._majority(targets[confidences >= high], self.high_action)
                    medium_mask = (confidences >= medium) & (confidences < high)
                    self.medium_action = self._majority(targets[medium_mask], self.medium_action)
                    self.low_action = self._majority(targets[confidences < medium], self.low_action)
        return self

    def decide(self, attack, confidence, severity):
        del attack, severity
        if confidence >= self.high_threshold:
            return self.high_action
        if confidence >= self.medium_threshold:
            return self.medium_action
        return self.low_action

    def _refit_actions(self, predictions, confidences, targets, medium, high):
        high_action = self._majority(targets[confidences >= high], self.high_action)
        medium_mask = (confidences >= medium) & (confidences < high)
        medium_action = self._majority(targets[medium_mask], self.medium_action)
        low_action = self._majority(targets[confidences < medium], self.low_action)
        return np.array(
            [high_action if c >= high else medium_action if c >= medium else low_action for c in confidences],
            dtype=object,
        )

    @staticmethod
    def _majority(values, fallback):
        if len(values) == 0:
            return fallback
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[int(np.argmax(counts))]
