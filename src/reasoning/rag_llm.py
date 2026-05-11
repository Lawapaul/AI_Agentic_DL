"""Retrieval-augmented reasoning strategy."""

from __future__ import annotations

from typing import Dict, List


KNOWLEDGE_BASE = {
    "DOS": "High-rate request bursts can degrade availability and overwhelm endpoints.",
    "DDOS": "Distributed traffic sources increase mitigation complexity and response urgency.",
    "PORTSCAN": "Sequential probing indicates reconnaissance preceding exploitation attempts.",
    "BRUTEFORCE": "Repeated authentication attempts signal credential abuse risk.",
    "BENIGN": "Traffic pattern does not match known malicious signatures strongly.",
}


class RAGReasoner:
    """Simple retrieval from internal attack knowledge snippets."""

    def _retrieve(self, attack_type: str) -> str:
        key = str(attack_type).upper()
        for name, text in KNOWLEDGE_BASE.items():
            if name in key or key in name:
                return text
        return "Attack class has limited indexed context; rely more on features and memory evidence."

    def explain(
        self,
        attack_type: str,
        confidence: float,
        risk_score: float,
        top_features: List[str],
        memory_context: Dict[str, object],
        graph_neighbors: List[str],
    ) -> Dict[str, str]:
        retrieved = self._retrieve(attack_type)
        features_text = ", ".join(top_features[:5]) if top_features else "none"

        return {
            "reasoning": (
                f"Retrieved knowledge: {retrieved} "
                f"Current evidence: confidence={confidence:.2f}, risk={risk_score:.2f}, features={features_text}."
            ),
            "attack_pattern": f"Knowledge-grounded interpretation for {attack_type} with graph links: {graph_neighbors[:3]}",
            "recommended_action": "Use retrieved prior plus current evidence to choose containment severity.",
            "impact": "Potential impact estimated from historical pattern frequency and current risk score.",
        }
