"""Rule-based decision policy engine."""

from __future__ import annotations

from typing import Dict


class RulePolicyPlanner:
    """Uses attack-type and risk-aware rules for action planning."""

    def decide(self, risk_score: float, attack_type: str, llm_explanation: Dict[str, str]) -> Dict[str, object]:
        del llm_explanation
        attack = str(attack_type).upper()
        risk = float(risk_score)

        if "DDOS" in attack or "DOS" in attack:
            action = "BLOCK_IP" if risk > 0.6 else "RATE_LIMIT"
            conf = 0.92
        elif "BRUTE" in attack or "SCAN" in attack:
            action = "RATE_LIMIT" if risk > 0.55 else "ALERT_ADMIN"
            conf = 0.86
        elif risk > 0.7:
            action, conf = "ALERT_ADMIN", 0.8
        else:
            action, conf = "ALLOW", 0.78

        return {
            "action": action,
            "confidence": conf,
            "reasoning": f"Rule engine selected {action} for attack={attack_type} risk={risk:.2f}.",
        }
