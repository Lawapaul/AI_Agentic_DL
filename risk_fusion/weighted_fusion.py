"""Static weighted adaptive risk fusion."""

from __future__ import annotations

from risk_fusion.utils import severity_from_risk, validate_signals


class RiskFusion:
    """Weighted risk fusion using fixed coefficients."""

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c, m, g, f = validate_signals(confidence, memory_sim, graph_sim, fg_strength)
        risk = 0.45 * c + 0.25 * m + 0.20 * g + 0.10 * f
        return {
            "risk_score": float(risk),
            "severity": severity_from_risk(risk),
        }
