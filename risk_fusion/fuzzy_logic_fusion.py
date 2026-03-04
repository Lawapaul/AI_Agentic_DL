"""Rule-based fuzzy logic adaptive risk fusion."""

from __future__ import annotations

from risk_fusion.utils import clamp01, severity_from_risk, validate_signals


class RiskFusion:
    """Simple fuzzy-rule fusion based on confidence, memory, and graph similarity."""

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c, m, g, f = validate_signals(confidence, memory_sim, graph_sim, fg_strength)

        high_conf = c > 0.8
        high_mem = m > 0.8
        high_graph = g > 0.8
        medium_conf = 0.5 <= c <= 0.8

        if high_conf and high_mem:
            risk = 0.90
        elif high_conf and high_graph:
            risk = 0.70
        elif medium_conf:
            risk = 0.50
        else:
            risk = 0.25

        # FG acts as a small modifier while keeping rule priority intact.
        risk = clamp01(risk + 0.05 * f)

        return {
            "risk_score": float(risk),
            "severity": severity_from_risk(risk),
        }
