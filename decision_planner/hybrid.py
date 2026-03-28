"""Hybrid decision planner with learned combination logic."""

from __future__ import annotations

import numpy as np

from decision_planner.confidence_based import ConfidenceBasedPlanner
from decision_planner.rule_based import RuleBasedPlanner


class HybridPlanner:
    def __init__(self):
        self.rule = RuleBasedPlanner()
        self.conf = ConfidenceBasedPlanner()
        self.block_threshold = 0.85
        self.override_action = "Block"

    def fit(self, attacks, confidences, severities, targets):
        self.rule.fit(attacks, confidences, targets)
        self.conf.fit(confidences, targets)

        attacks = np.asarray(attacks)
        confidences = np.asarray(confidences, dtype=float)
        severities = np.asarray(severities)
        targets = np.asarray(targets)

        best_score = -1.0
        for threshold in np.unique(confidences):
            high_mask = confidences >= threshold
            override_action = self._majority(targets[high_mask], self.override_action)
            predictions = np.array(
                [self._decide_with_params(a, c, s, threshold, override_action) for a, c, s in zip(attacks, confidences, severities)],
                dtype=object,
            )
            score = np.mean(predictions == targets)
            if score > best_score:
                best_score = score
                self.block_threshold = float(threshold)
                self.override_action = override_action
        return self

    def decide(self, attack, confidence, severity):
        return self._decide_with_params(attack, confidence, severity, self.block_threshold, self.override_action)

    def _decide_with_params(self, attack, confidence, severity, threshold, override_action):
        baseline = self.rule.decide(attack, confidence, severity)
        if baseline == self.rule.normal_action:
            return baseline
        if confidence >= threshold:
            return override_action
        return self.conf.decide(attack, confidence, severity)

    @staticmethod
    def _majority(values, fallback):
        if len(values) == 0:
            return fallback
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[int(np.argmax(counts))]
