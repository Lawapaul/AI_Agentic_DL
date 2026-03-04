from __future__ import annotations

from risk_fusion.utils import clamp01, severity_from_risk


class RiskFusion:
    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c = clamp01(float(confidence))
        m = clamp01(float(memory_sim))
        g = clamp01(float(graph_sim))
        f = clamp01(float(fg_strength))
        if c > 0.8 and m > 0.8:
            risk = 0.9
        elif c > 0.8 and g > 0.8:
            risk = 0.7
        elif 0.5 <= c <= 0.8:
            risk = 0.5
        else:
            risk = 0.25
        risk = clamp01(risk + 0.05 * f)
        return {"risk_score": float(risk), "severity": severity_from_risk(risk)}
