"""Lightweight RL-style decision agent for mitigation simulation."""

from __future__ import annotations

from typing import Dict

import numpy as np


class RLAgentPlanner:
    """Tabular Q-agent with pre-initialized policy over risk states."""

    ACTIONS = ("BLOCK_IP", "RATE_LIMIT", "ALERT_ADMIN", "ALLOW")

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        # State bins: low, medium, high, critical risk
        self.q_table = np.array(
            [
                [0.10, 0.15, 0.30, 0.70],
                [0.25, 0.45, 0.55, 0.20],
                [0.70, 0.65, 0.50, 0.10],
                [0.95, 0.70, 0.35, 0.05],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _state_index(risk_score: float) -> int:
        r = float(risk_score)
        if r > 0.8:
            return 3
        if r > 0.6:
            return 2
        if r > 0.4:
            return 1
        return 0

    def decide(self, risk_score: float, attack_type: str, llm_explanation: Dict[str, str]) -> Dict[str, object]:
        del attack_type, llm_explanation
        s = self._state_index(risk_score)
        q_values = self.q_table[s]
        a_idx = int(np.argmax(q_values))
        confidence = float(q_values[a_idx] / max(np.sum(q_values), 1e-6))
        return {
            "action": self.ACTIONS[a_idx],
            "confidence": confidence,
            "reasoning": "RL policy selected highest-value action for risk state.",
        }
