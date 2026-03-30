"""Independent runner for human review phase."""

from __future__ import annotations

from typing import Dict

from human_review.confidence_trigger import ConfidenceThresholdTrigger
from human_review.novelty_trigger import NoveltyTrigger
from human_review.uncertainty_trigger import UncertaintyTrigger


def run_human_review(sample_input: Dict[str, object], method: str = "confidence") -> Dict[str, object]:
    """Run standalone human-review trigger logic."""
    strategies = {
        "confidence": ConfidenceThresholdTrigger(),
        "uncertainty": UncertaintyTrigger(),
        "novelty": NoveltyTrigger(),
    }

    key = method.strip().lower()
    if key not in strategies:
        raise ValueError(f"Unknown human review method: {method}")

    return strategies[key].trigger(
        confidence=float(sample_input["confidence"]),
        risk_score=float(sample_input["risk_score"]),
        graph_distance=float(sample_input.get("graph_distance", 0.0)),
    )
