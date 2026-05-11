"""Risk-threshold decision planner."""

from __future__ import annotations

from typing import Dict


class RiskThresholdPlanner:
    """Maps risk scores to actions using fixed thresholds."""

    def decide(self, risk_score: float, attack_type: str, llm_explanation: Dict[str, str]) -> Dict[str, object]:
        del attack_type, llm_explanation
        r = float(risk_score)
        if r > 0.85:
            action, conf = "BLOCK_IP", 0.95
        elif r > 0.65:
            action, conf = "RATE_LIMIT", 0.85
        elif r > 0.45:
            action, conf = "ALERT_ADMIN", 0.75
        else:
            action, conf = "ALLOW", 0.8
        return {"action": action, "confidence": conf, "reasoning": "Threshold-based mitigation policy."}
