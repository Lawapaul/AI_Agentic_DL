"""Utility helpers for adaptive risk fusion methods."""

from __future__ import annotations

from typing import Tuple

import numpy as np


SEVERITY_LABELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def clamp01(value: float) -> float:
    """Clamp a numeric value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def validate_signals(
    confidence: float,
    memory_sim: float,
    graph_sim: float,
    fg_strength: float,
) -> Tuple[float, float, float, float]:
    """Validate and clamp all input risk signals to [0, 1]."""
    return (
        clamp01(confidence),
        clamp01(memory_sim),
        clamp01(graph_sim),
        clamp01(fg_strength),
    )


def severity_from_risk(risk_score: float) -> str:
    """Map risk score to severity class based on requested thresholds."""
    risk = clamp01(risk_score)
    if risk > 0.8:
        return "CRITICAL"
    if risk > 0.6:
        return "HIGH"
    if risk > 0.4:
        return "MEDIUM"
    return "LOW"


def severity_to_index(severity: str) -> int:
    """Convert severity string to ordinal index."""
    s = str(severity).strip().upper()
    if s not in SEVERITY_LABELS:
        raise ValueError(f"Unknown severity: {severity}")
    return SEVERITY_LABELS.index(s)


def index_to_severity(index: int) -> str:
    """Convert ordinal index to severity string."""
    i = int(index)
    i = max(0, min(i, len(SEVERITY_LABELS) - 1))
    return SEVERITY_LABELS[i]
