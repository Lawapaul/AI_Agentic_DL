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
        top_feature_names = [str(value).split(" (", 1)[0] for value in (top_features or [])[:3]]
        prompt = self.engine.generate_prompt(
            features,
            attack_name,
            confidence,
            risk_score=risk_score,
            top_features=top_feature_names,
            memory_similarity=memory_similarity,
            graph_weight=graph_weight,
            decision=decision,
        )
        output = self.generator.generate(prompt)
        if self._looks_like_instruction_echo(output):
            output = self._fallback_reasoning_text(
                attack_name=attack_name,
                risk_score=risk_score,
                top_features=top_feature_names,
                memory_similarity=memory_similarity,
                graph_weight=graph_weight,
                decision=decision,
            )
        return prompt, output

    @staticmethod
    def _looks_like_instruction_echo(text: str) -> bool:
        normalized = " ".join(str(text).strip().lower().split())
        if not normalized:
            return True
        bad_phrases = {
            "output exactly 2 to 4 sentences",
            "2 to 4 sentences",
            "recommended action sentence",
            "human-readable cybersecurity language",
        }
        return normalized in bad_phrases or normalized.startswith("output exactly")

    @staticmethod
    def _fallback_reasoning_text(
        attack_name: str,
        risk_score: float | None,
        top_features,
        memory_similarity: float | None,
        graph_weight: float | None,
        decision: str | None,
    ) -> str:
        if risk_score is None:
            risk_score = 0.5
        if memory_similarity is None:
            memory_similarity = 0.0
        if graph_weight is None:
            graph_weight = 0.0
        feature_text = ", ".join(top_features[:2]) if top_features else "the most important network features"
        action_text = str(decision or "Monitor")
        risk_text = "High" if float(risk_score) >= 0.75 else "Medium" if float(risk_score) >= 0.4 else "Low"
        return (
            f"{risk_text} risk detected for {attack_name.lower()} traffic because the sample shows strong memory similarity "
            f"and graph correlation with previously observed behavior. Key contributing features include {feature_text}. "
            f"Recommended action: {action_text}."
        )
