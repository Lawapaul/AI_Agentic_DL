"""Risk-based decision planner with learned weights and thresholds."""

from __future__ import annotations

import numpy as np


class RiskBasedPlanner:
    def __init__(self):
        self.severity_map = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
        self.confidence_weight = 0.6
        self.severity_weight = 0.4
        self.block_threshold = 0.8
        self.alert_threshold = 0.5
        self.block_action = "Block"
        self.alert_action = "Alert"
        self.monitor_action = "Monitor"

    def fit(self, confidences, severities, targets):
        confidences = np.asarray(confidences, dtype=float)
        severities = np.asarray(severities)
        severity_scores = np.asarray([self.severity_map.get(s, 0.5) for s in severities], dtype=float)
        targets = np.asarray(targets)
        best_score = -1.0

        for confidence_weight in np.linspace(0.2, 0.8, 7):
            severity_weight = 1.0 - confidence_weight
            risks = (confidence_weight * confidences) + (severity_weight * severity_scores)
            thresholds = np.unique(risks)
            for alert_threshold in thresholds:
                for block_threshold in thresholds:
                    if block_threshold < alert_threshold:
                        continue
                    high_mask = risks >= block_threshold
                    med_mask = (risks >= alert_threshold) & (risks < block_threshold)
                    low_mask = risks < alert_threshold
                    block_action = self._majority(targets[high_mask], self.block_action)
                    alert_action = self._majority(targets[med_mask], self.alert_action)
                    monitor_action = self._majority(targets[low_mask], self.monitor_action)
                    predictions = np.where(
                        high_mask,
                        block_action,
                        np.where(med_mask, alert_action, monitor_action),
                    )
                    score = np.mean(predictions == targets)
                    if score > best_score:
                        best_score = score
                        self.confidence_weight = float(confidence_weight)
                        self.severity_weight = float(severity_weight)
                        self.alert_threshold = float(alert_threshold)
                        self.block_threshold = float(block_threshold)
                        self.block_action = block_action
                        self.alert_action = alert_action
                        self.monitor_action = monitor_action
        return self

    def decide(self, attack, confidence, severity):
        del attack
        risk = (self.confidence_weight * confidence) + (
            self.severity_weight * self.severity_map.get(severity, 0.5)
        )
        if risk >= self.block_threshold:
            return self.block_action
        if risk >= self.alert_threshold:
            return self.alert_action
        return self.monitor_action

    @staticmethod
    def _majority(values, fallback):
        if len(values) == 0:
            return fallback
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[int(np.argmax(counts))]
