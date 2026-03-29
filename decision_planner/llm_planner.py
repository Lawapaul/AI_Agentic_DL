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

        self.task = "text2text-generation"
        try:
            self.generator = pipeline("text2text-generation", model=model)
        except Exception:
            self.task = "text-generation"
            self.generator = pipeline("text-generation", model="distilgpt2")
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

        if self.task == "text2text-generation":
            result = self.generator(prompt, do_sample=False, max_new_tokens=10)
            return result[0]["generated_text"].strip()

        result = self.generator(
            f"{prompt}\nAction:",
            do_sample=False,
            max_new_tokens=6,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        generated = result[0]["generated_text"].split("Action:", 1)[-1].strip()
        action = generated.split()[0] if generated else "Monitor"
        if action not in {"No", "Monitor", "Alert", "Block"}:
            return "Monitor"
        return "No Action" if action == "No" else action
