"""Probabilistic decision planner."""

from __future__ import annotations

from typing import Dict

import numpy as np


class ProbabilisticPlanner:
    """Samples action from risk-calibrated probability distribution."""

    ACTIONS = ("BLOCK_IP", "RATE_LIMIT", "ALERT_ADMIN", "ALLOW")

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def decide(self, risk_score: float, attack_type: str, llm_explanation: Dict[str, str]) -> Dict[str, object]:
        del attack_type, llm_explanation
        r = float(np.clip(risk_score, 0.0, 1.0))
        probs = np.array([
            0.05 + 0.70 * r,
            0.15 + 0.35 * r,
            0.35 - 0.10 * r,
            0.45 - 0.95 * r,
        ])
        probs = np.clip(probs, 1e-6, None)
        probs = probs / probs.sum()

        idx = int(self.rng.choice(len(self.ACTIONS), p=probs))
        action = self.ACTIONS[idx]
        return {
            "action": action,
            "confidence": float(probs[idx]),
            "reasoning": "Action sampled from calibrated mitigation distribution.",
        }
