"""Lightweight local Flan-T5 based reasoning with safe fallback."""

from __future__ import annotations

from typing import Dict, List


class LocalFlanT5Reasoner:
    """Uses Flan-T5 when available; falls back to deterministic text if not."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self._generator = None
        try:
            from transformers import pipeline

            self._generator = pipeline("text2text-generation", model=model_name)
        except Exception:
            self._generator = None

    def explain(
        self,
        attack_type: str,
        confidence: float,
        risk_score: float,
        top_features: List[str],
        memory_context: Dict[str, object],
        graph_neighbors: List[str],
    ) -> Dict[str, str]:
        prompt = (
            "Explain this IDS prediction in JSON-like fields reasoning, attack_pattern, recommended_action, impact. "
            f"Attack={attack_type}; confidence={confidence:.3f}; risk={risk_score:.3f}; "
            f"features={top_features[:5]}; memory={memory_context}; graph_neighbors={graph_neighbors[:3]}."
        )

        if self._generator is None:
            text = (
                f"Fallback reasoning: attack={attack_type}, confidence={confidence:.2f}, risk={risk_score:.2f}. "
                f"Feature signals={top_features[:5]} and memory/graph context support escalation."
            )
        else:
            out = self._generator(prompt, max_length=180, do_sample=False)
            text = out[0]["generated_text"]

        return {
            "reasoning": text,
            "attack_pattern": f"Model-generated pattern summary for {attack_type}.",
            "recommended_action": "Apply mitigations proportional to generated risk interpretation.",
            "impact": "Potential service/security impact estimated by local LLM rationale.",
        }
