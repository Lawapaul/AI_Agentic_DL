"""End-to-end lightweight LLM reasoning pipeline."""

from llm_reasoning.llm_engine import LLMReasoningEngine
from llm_reasoning.reasoning_generator import ReasoningGenerator


class LLMPipeline:
    """Generate prompts and model reasoning for one IDS sample."""

    def __init__(self):
        self.engine = LLMReasoningEngine()
        self.generator = ReasoningGenerator()

    def run(self, features, prediction, confidence: float):
        prompt = self.engine.generate_prompt(features, prediction, confidence)
        output = self.generator.generate(prompt)
        return prompt, output
