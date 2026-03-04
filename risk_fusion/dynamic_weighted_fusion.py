from __future__ import annotations

from risk_fusion.utils import severity_from_risk


class RiskFusion:
    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c = max(0.0, min(1.0, float(confidence)))
        m = max(0.0, min(1.0, float(memory_sim)))
        g = max(0.0, min(1.0, float(graph_sim)))
        f = max(0.0, min(1.0, float(fg_strength)))
        if c > 0.9:
            w = (0.6, 0.2, 0.1, 0.1)
        else:
            w = (0.35, 0.3, 0.25, 0.1)
        risk = w[0] * c + w[1] * m + w[2] * g + w[3] * f
        return {"risk_score": float(risk), "severity": severity_from_risk(risk)}
