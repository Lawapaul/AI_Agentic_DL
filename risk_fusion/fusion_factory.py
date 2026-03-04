"""Factory for adaptive risk fusion strategies."""

from __future__ import annotations

from risk_fusion.attention_fusion import RiskFusion as AttentionFusion
from risk_fusion.dynamic_weighted_fusion import RiskFusion as DynamicWeightedFusion
from risk_fusion.fuzzy_logic_fusion import RiskFusion as FuzzyLogicFusion
from risk_fusion.logistic_regression_fusion import RiskFusion as LogisticRegressionFusion
from risk_fusion.random_forest_fusion import RiskFusion as RandomForestFusion
from risk_fusion.weighted_fusion import RiskFusion as WeightedFusion


class FusionFactory:
    """Create risk fusion strategy instances by method name."""

    _METHODS = {
        "weighted": WeightedFusion,
        "dynamic_weighted": DynamicWeightedFusion,
        "logistic_regression": LogisticRegressionFusion,
        "random_forest": RandomForestFusion,
        "fuzzy_logic": FuzzyLogicFusion,
        "attention": AttentionFusion,
    }

    @classmethod
    def create(cls, method: str = "random_forest"):
        method_key = str(method).strip().lower()
        if method_key not in cls._METHODS:
            available = ", ".join(sorted(cls._METHODS))
            raise ValueError(f"Unknown fusion method: {method}. Available: {available}")
        return cls._METHODS[method_key]()
