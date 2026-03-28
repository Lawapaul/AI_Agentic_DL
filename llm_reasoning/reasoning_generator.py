"""Transformer-backed text generator for IDS reasoning."""

from __future__ import annotations

from transformers import pipeline


class ReasoningGenerator:
    """Wrap a small Hugging Face text2text model for concise reasoning."""

    def __init__(self):
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=200,
        )

    def generate(self, prompt: str) -> str:
        result = self.generator(prompt, do_sample=False)
        return result[0]["generated_text"]
