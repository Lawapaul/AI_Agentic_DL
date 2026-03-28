"""LLM-backed decision planner."""

from transformers import pipeline


class LLMPlanner:
    def __init__(self, model: str = "google/flan-t5-small"):
        self.generator = pipeline("text2text-generation", model=model)

    def decide(self, attack, confidence, severity):
        prompt = f"""
Attack: {attack}
Confidence: {confidence:.2f}
Severity: {severity}

Choose best action: Block / Monitor / Alert
Output only one word.
""".strip()
        result = self.generator(prompt, max_length=20)
        return result[0]["generated_text"].strip()
