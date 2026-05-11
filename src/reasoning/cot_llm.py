"""Chain-of-thought style reasoning generator."""

from __future__ import annotations

from typing import Dict, List


class ChainOfThoughtReasoner:
    """Produces concise stepwise reasoning without exposing long hidden traces."""

    def explain(
        self,
        attack_type: str,
        confidence: float,
        risk_score: float,
        top_features: List[str],
        memory_context: Dict[str, object],
        graph_neighbors: List[str],
    ) -> Dict[str, str]:
        features_text = ", ".join(top_features[:4]) if top_features else "weak feature signal"
        memory_hint = memory_context.get("summary", "memory context unavailable")
        neighbors_text = ", ".join(graph_neighbors[:3]) if graph_neighbors else "none"

        reasoning = (
            "Step 1: Validate classifier output against confidence and risk. "
            f"Observed attack={attack_type}, confidence={confidence:.2f}, risk={risk_score:.2f}. "
            f"Step 2: Cross-check top features ({features_text}) with stored patterns ({memory_hint}). "
            f"Step 3: Verify graph neighborhood overlap ({neighbors_text}) to estimate propagation potential. "
            "Step 4: Recommend containment proportional to aggregated evidence."
        )

        return {
            "reasoning": reasoning,
            "attack_pattern": f"Evidence chain supports {attack_type} behavioral signature.",
            "recommended_action": "Apply progressive controls and continue context-aware monitoring.",
            "impact": "Likely operational impact depends on lateral movement and persistence probability.",
        }
