"""Transformer-backed text generator for IDS reasoning."""

from __future__ import annotations

from transformers import pipeline


class ReasoningGenerator:
    """Wrap a small Hugging Face text2text model for concise reasoning."""

    def __init__(self, model: str = "google/flan-t5-small"):
        self.generator = pipeline(
            "text2text-generation",
            model=model,
        )

    def generate(self, prompt: str) -> str:
        result = self.generator(
            prompt,
            do_sample=True,
            temperature=0.7,
            max_length=200,
        )
        return result[0]["generated_text"]
