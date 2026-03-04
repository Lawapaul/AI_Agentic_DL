"""Attention-based adaptive risk fusion using PyTorch."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from risk_fusion.utils import clamp01, severity_from_risk, validate_signals


class _AttentionFusionNet(nn.Module):
    """Tiny attention model over [C, M, G, F] signals."""

    def __init__(self):
        super().__init__()
        self.attn_layer = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor):
        logits = self.attn_layer(x)
        weights = torch.softmax(logits, dim=-1)
        weighted_sum = torch.sum(weights * x, dim=-1, keepdim=True)
        risk = torch.sigmoid(weighted_sum)
        return risk, weights


class RiskFusion:
    """Attention-based risk fusion with interpretable per-signal weights."""

    def __init__(self, random_state: int = 42, epochs: int = 200, lr: float = 0.01):
        self.random_state = random_state
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _AttentionFusionNet().to(self.device)
        self._is_fitted = False
        self._last_attention: List[float] = [0.25, 0.25, 0.25, 0.25]

    def fit(self, X, y):
        """Train attention fusion model on [C, M, G, F] to predict severity risk."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != 4:
            raise ValueError("X must be shape (N, 4) with [C, M, G, F].")

        # Severity classes 0..3 mapped to continuous target risk in [0,1].
        y_target = np.clip(y_arr / 3.0, 0.0, 1.0).reshape(-1, 1)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_target, dtype=torch.float32, device=self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds, _ = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        self._is_fitted = True
        return self

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c, m, g, f = validate_signals(confidence, memory_sim, graph_sim, fg_strength)

        if not self._is_fitted:
            risk = 0.45 * c + 0.25 * m + 0.20 * g + 0.10 * f
            return {
                "risk_score": float(risk),
                "severity": severity_from_risk(risk),
                "attention_weights": [0.25, 0.25, 0.25, 0.25],
            }

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([[c, m, g, f]], dtype=torch.float32, device=self.device)
            risk_tensor, weights = self.model(x)
            risk = clamp01(float(risk_tensor.squeeze().cpu().item()))
            attn = [float(v) for v in weights.squeeze(0).cpu().numpy().tolist()]
            self._last_attention = attn

        return {
            "risk_score": float(risk),
            "severity": severity_from_risk(risk),
            "attention_weights": self._last_attention,
        }
