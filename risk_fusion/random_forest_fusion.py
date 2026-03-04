from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from risk_fusion.utils import clamp01, severity_from_risk


class RiskFusion:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self.model.fit(X, y)
        self.fitted = True

    def compute_risk(self, confidence, memory_sim, graph_sim, fg_strength):
        c, m, g, f = [clamp01(v) for v in [confidence, memory_sim, graph_sim, fg_strength]]
        if not self.fitted:
            risk = 0.45 * c + 0.25 * m + 0.2 * g + 0.1 * f
            return {"risk_score": float(risk), "severity": severity_from_risk(risk)}
        proba = self.model.predict_proba(np.array([[c, m, g, f]], dtype=np.float32))[0]
        cls = self.model.classes_.astype(np.float32)
        risk = clamp01(float(np.dot(proba, cls) / max(cls.max(), 1.0)))
        return {"risk_score": float(risk), "severity": severity_from_risk(risk)}
