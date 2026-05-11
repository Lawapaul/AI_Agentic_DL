"""Confidence-threshold human review trigger."""

from __future__ import annotations

from typing import Dict


class ConfidenceThresholdTrigger:
    """Escalates to analyst when model confidence is low."""

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence

    def trigger(self, confidence: float, risk_score: float, graph_distance: float) -> Dict[str, object]:
        del risk_score, graph_distance
        if float(confidence) < self.min_confidence:
            return {"review_required": True, "reason": "Low classifier confidence"}
        return {"review_required": False, "reason": "Confidence above threshold"}
