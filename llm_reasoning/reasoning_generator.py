"""Transformer-backed text generator for IDS reasoning."""

from __future__ import annotations

from transformers import pipeline


class ReasoningGenerator:
    """Wrap a small Hugging Face text2text model for concise reasoning."""

    def __init__(self, model: str = "google/flan-t5-small"):
        self.task = "text2text-generation"
        self.prompt_echo = False
        try:
            self.generator = pipeline(
                "text2text-generation",
                model=model,
            )
        except Exception:
            # Colab images occasionally ship a transformers build without the
            # text2text alias. Fall back to a lightweight causal LM instead.
            self.task = "text-generation"
            self.prompt_echo = True
            self.generator = pipeline(
                "text-generation",
                model="distilgpt2",
            )

    def generate(self, prompt: str) -> str:
        if self.task == "text2text-generation":
            result = self.generator(
                prompt,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=48,
            )
            return result[0]["generated_text"].strip()

        reasoning_prompt = f"{prompt}\nShort reasoning:"
        result = self.generator(
            reasoning_prompt,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=48,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        generated = result[0]["generated_text"]
        if self.prompt_echo and generated.startswith(reasoning_prompt):
            generated = generated[len(reasoning_prompt):]
        return generated.strip() or "Suspicious pattern detected from model, graph, and memory signals."
