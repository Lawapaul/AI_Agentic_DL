"""Rule-based template reasoning for IDS outputs."""

from __future__ import annotations

from typing import Dict, List


class RuleBasedReasoner:
    """Deterministic reasoning strategy based on templates."""

    def explain(
        self,
        attack_type: str,
        confidence: float,
        risk_score: float,
        top_features: List[str],
        memory_context: Dict[str, object],
        graph_neighbors: List[str],
    ) -> Dict[str, str]:
        confidence_band = "high" if confidence >= 0.8 else "moderate" if confidence >= 0.5 else "low"
        risk_band = "critical" if risk_score > 0.8 else "high" if risk_score > 0.6 else "medium" if risk_score > 0.4 else "low"
        features_text = ", ".join(top_features[:5]) if top_features else "limited salient indicators"
        neighbors_text = ", ".join(graph_neighbors[:3]) if graph_neighbors else "no strongly correlated classes"

        return {
            "reasoning": (
                f"Model indicates {attack_type} with {confidence_band} confidence and {risk_band} risk. "
                f"Key contributors include {features_text}. Memory context suggests {memory_context.get('summary', 'no close historical match')}."
            ),
            "attack_pattern": f"Pattern aligns with {attack_type}; correlated graph neighbors: {neighbors_text}.",
            "recommended_action": "Escalate mitigation based on risk tier and monitor traffic segment.",
            "impact": f"Potential impact is {risk_band} on service availability and incident response load.",
        }
