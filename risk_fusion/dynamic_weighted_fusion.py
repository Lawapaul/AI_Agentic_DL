"""Dynamic weighted adaptive risk fusion."""

from __future__ import annotations

from risk_fusion.utils import severity_from_risk, validate_signals


class RiskFusion:
    """Weighted risk fusion with confidence-dependent weights."""

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c, m, g, f = validate_signals(confidence, memory_sim, graph_sim, fg_strength)

        if c > 0.9:
            w_c, w_m, w_g, w_f = (0.6, 0.2, 0.1, 0.1)
        else:
            w_c, w_m, w_g, w_f = (0.35, 0.3, 0.25, 0.1)

        risk = w_c * c + w_m * m + w_g * g + w_f * f
        return {
            "risk_score": float(risk),
            "severity": severity_from_risk(risk),
        }
