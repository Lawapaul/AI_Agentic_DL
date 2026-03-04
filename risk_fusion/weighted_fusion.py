from __future__ import annotations

from risk_fusion.utils import severity_from_risk


class RiskFusion:
    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c = max(0.0, min(1.0, float(confidence)))
        m = max(0.0, min(1.0, float(memory_sim)))
        g = max(0.0, min(1.0, float(graph_sim)))
        f = max(0.0, min(1.0, float(fg_strength)))
        risk = 0.45 * c + 0.25 * m + 0.2 * g + 0.1 * f
        return {"risk_score": float(risk), "severity": severity_from_risk(risk)}
