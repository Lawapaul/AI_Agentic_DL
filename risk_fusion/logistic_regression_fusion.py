"""Logistic Regression adaptive risk fusion."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from risk_fusion.utils import clamp01, severity_from_risk, validate_signals


class RiskFusion:
    """Severity-aware fusion model using Logistic Regression."""

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y):
        """Train logistic regression on [C, M, G, F] features and severity labels."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        if X_arr.ndim != 2 or X_arr.shape[1] != 4:
            raise ValueError("X must be shape (N, 4) with [C, M, G, F].")
        self.model.fit(X_arr, y_arr)
        self._is_fitted = True
        return self

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c, m, g, f = validate_signals(confidence, memory_sim, graph_sim, fg_strength)

        if not self._is_fitted:
            # Deterministic fallback before training.
            risk = 0.45 * c + 0.25 * m + 0.20 * g + 0.10 * f
            return {"risk_score": float(risk), "severity": severity_from_risk(risk)}

        X = np.array([[c, m, g, f]], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]
        class_values = np.asarray(self.model.classes_, dtype=np.float32)

        # Expected severity class index mapped to [0, 1] risk.
        expected_class = float(np.dot(proba, class_values))
        max_class = float(max(class_values.max(), 1.0))
        risk = clamp01(expected_class / max_class)

        return {
            "risk_score": float(risk),
            "severity": severity_from_risk(risk),
        }
