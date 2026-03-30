"""Independent pipeline runner for LLM reasoning strategies."""

from __future__ import annotations

from typing import Dict

from llm_reasoning.cot_llm import ChainOfThoughtReasoner
from llm_reasoning.local_flan_t5 import LocalFlanT5Reasoner
from llm_reasoning.rag_llm import RAGReasoner
from llm_reasoning.rule_based_llm import RuleBasedReasoner


def run_llm_reasoning(sample_input: Dict[str, object], method: str = "rule_based") -> Dict[str, str]:
    """Run standalone LLM reasoning phase for one sample."""
    strategies = {
        "rule_based": RuleBasedReasoner(),
        "cot": ChainOfThoughtReasoner(),
        "rag": RAGReasoner(),
        "flan_t5": LocalFlanT5Reasoner(),
    }

    key = method.strip().lower()
    if key not in strategies:
        raise ValueError(f"Unknown LLM reasoning method: {method}")

    reasoner = strategies[key]
    return reasoner.explain(
        attack_type=str(sample_input["attack_type"]),
        confidence=float(sample_input["confidence"]),
        risk_score=float(sample_input["risk_score"]),
        top_features=list(sample_input.get("top_features", [])),
        memory_context=dict(sample_input.get("memory_context", {})),
        graph_neighbors=list(sample_input.get("graph_neighbors", [])),
    )
