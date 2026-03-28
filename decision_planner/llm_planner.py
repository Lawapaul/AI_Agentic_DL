"""LLM-backed decision planner."""

from __future__ import annotations


class LLMPlanner:
    def __init__(self, model: str = "google/flan-t5-small"):
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LLMPlanner could not import transformers. Reinstall compatible Colab "
                "dependencies and restart the runtime before using LLMPlanner."
            ) from exc

        self.generator = pipeline("text2text-generation", model=model)
        self.examples = []

    def fit(self, attacks, confidences, severities, targets, max_examples: int = 4):
        self.examples = []
        for attack, confidence, severity, target in zip(attacks, confidences, severities, targets):
            if len(self.examples) >= max_examples:
                break
            self.examples.append(
                {
                    "attack": str(attack),
                    "confidence": float(confidence),
                    "severity": str(severity),
                    "action": str(target),
                }
            )
        return self

    def decide(self, attack, confidence, severity):
        if attack == "Normal Traffic":
            return "No Action"

        prompt = f"""
You are a cybersecurity system.

STRICT RULE:
If attack is "Normal Traffic" -> output ONLY "No Action"

Input:
Attack: {attack}
Confidence: {confidence:.2f}
Severity: {severity}

Choose ONE action:
No Action / Monitor / Alert / Block

Output ONLY one word.
""".strip()

        result = self.generator(prompt, do_sample=False, max_length=20)
        return result[0]["generated_text"].strip()
