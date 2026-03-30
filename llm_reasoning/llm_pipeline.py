"""End-to-end lightweight LLM reasoning pipeline."""

from llm_reasoning.llm_engine import LLMReasoningEngine
from llm_reasoning.reasoning_generator import ReasoningGenerator


LABEL_MAP = {
    0: "Normal Traffic",
    1: "DoS Attack",
    2: "Port Scan",
    3: "Brute Force Attack",
    4: "Web Attack",
    5: "Botnet",
    6: "Infiltration",
}


class LLMPipeline:
    """Generate prompts and model reasoning for one IDS sample."""

    def __init__(self, model: str = "google/flan-t5-small"):
        self.engine = LLMReasoningEngine()
        self.generator = ReasoningGenerator(model=model)

    def run(
        self,
        features,
        prediction,
        confidence: float,
        risk_score: float | None = None,
        top_features=None,
        memory_similarity: float | None = None,
        graph_weight: float | None = None,
        decision: str | None = None,
    ):
        attack_name = LABEL_MAP.get(int(prediction), "Unknown Attack")
        prompt = self.engine.generate_prompt(
            features,
            attack_name,
            confidence,
            risk_score=risk_score,
            top_features=top_features,
            memory_similarity=memory_similarity,
            graph_weight=graph_weight,
            decision=decision,
        )
        output = self.generator.generate(prompt)
        return prompt, output
