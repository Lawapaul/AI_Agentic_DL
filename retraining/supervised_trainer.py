"""Supervised trainer for learning human-corrected decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from retraining.data_loader import ACTIONS


LOGGER = logging.getLogger(__name__)


class DecisionMLP(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, output_dim: int = len(ACTIONS)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TrainingSummary:
    epochs: int
    loss: float
    samples: int


class SupervisedDecisionTrainer:
    def __init__(self):
        self.model = DecisionMLP()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_model(self, features: torch.Tensor, labels: torch.Tensor, epochs: int = 20) -> TrainingSummary | None:
        if len(features) == 0:
            LOGGER.warning("Skipping supervised training because no feedback samples are available")
            return None

        self.model.train()
        loss_value = 0.0
        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().cpu().item())

        LOGGER.info("Supervised trainer updated on %d samples; loss=%.4f", len(features), loss_value)
        return TrainingSummary(epochs=epochs, loss=loss_value, samples=int(len(features)))

    def predict_action(self, features: torch.Tensor) -> str:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features.unsqueeze(0))
            index = int(torch.argmax(logits, dim=1).item())
        return ACTIONS[index]
