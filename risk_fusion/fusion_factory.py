from __future__ import annotations

from risk_fusion.attention_fusion import RiskFusion as Attention
from risk_fusion.dynamic_weighted_fusion import RiskFusion as Dynamic
from risk_fusion.fuzzy_logic_fusion import RiskFusion as Fuzzy
from risk_fusion.logistic_regression_fusion import RiskFusion as LR
from risk_fusion.random_forest_fusion import RiskFusion as RF
from risk_fusion.weighted_fusion import RiskFusion as Weighted


class FusionFactory:
    @staticmethod
    def create(method: str):
        m = method.strip().lower()
        mapping = {
            "weighted": Weighted,
            "dynamic_weighted": Dynamic,
            "logistic_regression": LR,
            "random_forest": RF,
            "fuzzy_logic": Fuzzy,
            "attention": Attention,
        }
        if m not in mapping:
            raise ValueError(f"Unknown method: {method}")
        return mapping[m]()
