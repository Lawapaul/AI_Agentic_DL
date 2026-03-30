"""LLM-backed decision planner."""

from __future__ import annotations

import torch


class LLMPlanner:
    def __init__(self, model: str = "google/flan-t5-small"):
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LLMPlanner could not import transformers. Reinstall compatible Colab "
                "dependencies and restart the runtime before using LLMPlanner."
            ) from exc

        self.task = "seq2seq"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        except Exception:
            self.task = "causal"
            fallback_model = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
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

        generation_prompt = prompt if self.task == "seq2seq" else f"{prompt}\nAction:"
        inputs = self.tokenizer(
            generation_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=10 if self.task == "seq2seq" else 6,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if self.task == "causal" and generated.startswith(generation_prompt):
            generated = generated[len(generation_prompt):]
        generated = generated.split("Action:", 1)[-1].strip()
        action = generated.split()[0] if generated else "Monitor"
        if action not in {"No", "Monitor", "Alert", "Block"}:
            return "Monitor"
        return "No Action" if action == "No" else action
