"""Utilities for risk fusion outputs."""

from __future__ import annotations

import numpy as np


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def severity_from_risk(risk: float) -> str:
    r = clamp01(risk)
    if r > 0.8:
        return "CRITICAL"
    if r > 0.6:
        return "HIGH"
    if r > 0.4:
        return "MEDIUM"
    return "LOW"


def severity_to_idx(sev: str) -> int:
    labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    return labels.index(str(sev).upper())
