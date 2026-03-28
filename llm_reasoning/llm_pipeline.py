"""End-to-end lightweight LLM reasoning pipeline."""

from __future__ import annotations

from .llm_engine import LLMReasoningEngine
from .reasoning_generator import ReasoningGenerator


class LLMPipeline:
    """Generate prompts and model reasoning for one IDS sample."""

    def __init__(self, model_name: str = "google/flan-t5-small", max_length: int = 200):
        self.engine = LLMReasoningEngine()
        self.generator = ReasoningGenerator(model_name=model_name, max_length=max_length)

    def run(self, features, prediction, confidence: float):
        prompt = self.engine.generate_prompt(features, prediction, confidence)
        output = self.generator.generate(prompt)
        return prompt, output
