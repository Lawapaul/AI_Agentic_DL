"""LLM-backed decision planner with learned in-context examples."""

from __future__ import annotations

from transformers import pipeline


class LLMPlanner:
    def __init__(self, model: str = "google/flan-t5-small"):
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
        example_block = ""
        if self.examples:
            lines = []
            for item in self.examples:
                lines.append(
                    f"Attack: {item['attack']}\nConfidence: {item['confidence']:.2f}\nSeverity: {item['severity']}\nBest Action: {item['action']}"
                )
            example_block = "Examples:\n" + "\n\n".join(lines) + "\n\n"

        prompt = f"""
{example_block}Attack: {attack}
Confidence: {confidence:.2f}
Severity: {severity}

Choose the best action from: Block, Monitor, Alert, No Action, Log Only, Alert + Monitor.
Output only the action phrase.
""".strip()
        result = self.generator(prompt, max_length=24)
        return result[0]["generated_text"].strip()
