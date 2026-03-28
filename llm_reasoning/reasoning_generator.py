"""Transformer-backed text generator for IDS reasoning."""

from __future__ import annotations

from transformers import pipeline


class ReasoningGenerator:
    """Wrap a small Hugging Face text2text model for concise reasoning."""

    def __init__(self, model_name: str = "google/flan-t5-small", max_length: int = 200):
        self.model_name = model_name
        self.max_length = max_length
        self.generator = pipeline(
            "text2text-generation",
            model=self.model_name,
            max_length=self.max_length,
        )

    def generate(self, prompt: str) -> str:
        result = self.generator(prompt)
        return result[0]["generated_text"].strip()
