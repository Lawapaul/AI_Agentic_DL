from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from risk_fusion.utils import clamp01, severity_from_risk


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        w = torch.softmax(self.fc(x), dim=-1)
        risk = torch.sigmoid(torch.sum(w * x, dim=-1, keepdim=True))
        return risk, w


class RiskFusion:
    def __init__(self, random_state: int = 42, epochs: int = 100, lr: float = 0.01):
        self.model = _Attn()
        self.epochs = epochs
        self.lr = lr
        self.fitted = False
        torch.manual_seed(random_state)

    def fit(self, X, y):
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        y = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1) / 3.0)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred, _ = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        self.fitted = True

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        x = torch.tensor([[clamp01(confidence), clamp01(memory_sim), clamp01(graph_sim), clamp01(fg_strength)]], dtype=torch.float32)
        if not self.fitted:
            c, m, g, f = x.numpy()[0]
            risk = 0.45 * c + 0.25 * m + 0.2 * g + 0.1 * f
            return {"risk_score": float(risk), "severity": severity_from_risk(risk), "attention_weights": [0.25, 0.25, 0.25, 0.25]}
        self.model.eval()
        with torch.no_grad():
            r, w = self.model(x)
        risk = clamp01(float(r.item()))
        return {"risk_score": float(risk), "severity": severity_from_risk(risk), "attention_weights": [float(v) for v in w.numpy()[0].tolist()]}
