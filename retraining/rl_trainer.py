"""Simple Q-learning trainer for adaptive action selection."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger(__name__)


class RLTrainer:
    def __init__(self, actions: Sequence[str] | None = None):
        self.actions = list(actions or ["No Action", "Monitor", "Alert", "Block"])
        self.q_table: dict[str, dict[str, float]] = {}

    def _state_key(self, risk_score: float, context_embedding: Sequence[float] | None = None) -> str:
        risk_bucket = min(int(float(risk_score) * 10), 9)
        context_bucket = 0
        if context_embedding:
            context_bucket = min(int(abs(float(context_embedding[0])) * 10), 9)
        return f"risk_{risk_bucket}_ctx_{context_bucket}"

    def select_action(
        self,
        risk_score: float,
        context_embedding: Sequence[float] | None = None,
        epsilon: float = 0.05,
    ) -> str:
        state = self._state_key(risk_score, context_embedding)
        self.q_table.setdefault(state, {action: 0.0 for action in self.actions})
        if random.random() < epsilon:
            action = random.choice(self.actions)
            LOGGER.info("RL explorer selected %s for %s", action, state)
            return action
        action = max(self.q_table[state], key=self.q_table[state].get)
        LOGGER.info("RL policy selected %s for %s", action, state)
        return action

    def update_policy(
        self,
        risk_score: float,
        action: str,
        reward: float,
        context_embedding: Sequence[float] | None = None,
        alpha: float = 0.2,
    ) -> None:
        state = self._state_key(risk_score, context_embedding)
        self.q_table.setdefault(state, {candidate: 0.0 for candidate in self.actions})
        current = self.q_table[state].get(action, 0.0)
        self.q_table[state][action] = current + alpha * (float(reward) - current)
        LOGGER.info("Updated RL policy for %s/%s to %.3f", state, action, self.q_table[state][action])

    def dump_policy(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.q_table, indent=2))
