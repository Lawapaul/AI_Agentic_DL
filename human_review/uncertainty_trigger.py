"""Risk uncertainty based human review trigger."""

from __future__ import annotations

from typing import Dict


class UncertaintyTrigger:
    """Escalates cases near uncertain risk zone for analyst review."""

    def __init__(self, lower: float = 0.45, upper: float = 0.65):
        self.lower = lower
        self.upper = upper

    def trigger(self, confidence: float, risk_score: float, graph_distance: float) -> Dict[str, object]:
        del confidence, graph_distance
        r = float(risk_score)
        if self.lower <= r <= self.upper:
            return {"review_required": True, "reason": "Risk in uncertainty band"}
        return {"review_required": False, "reason": "Risk outside uncertainty band"}
