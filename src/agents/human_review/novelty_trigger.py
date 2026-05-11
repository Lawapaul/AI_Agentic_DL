"""Novel attack trigger using graph distance."""

from __future__ import annotations

from typing import Dict


class NoveltyTrigger:
    """Escalates potentially novel samples based on graph distance."""

    def __init__(self, novelty_threshold: float = 0.6):
        self.novelty_threshold = novelty_threshold

    def trigger(self, confidence: float, risk_score: float, graph_distance: float) -> Dict[str, object]:
        del confidence, risk_score
        d = float(graph_distance)
        if d > self.novelty_threshold:
            return {"review_required": True, "reason": "Novel pattern by graph distance"}
        return {"review_required": False, "reason": "Graph-near known attack clusters"}
