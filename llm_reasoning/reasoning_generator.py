"""Transformer-backed text generator for IDS reasoning."""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


class ReasoningGenerator:
    """Wrap a small Hugging Face text2text model for concise reasoning."""

    def __init__(self, model: str = "google/flan-t5-small"):
        self.task = "seq2seq"
        self.prompt_echo = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
            if hasattr(self.model, "config"):
                self.model.config.tie_word_embeddings = False
        except Exception:
            self.task = "causal"
            self.prompt_echo = True
            fallback_model = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @staticmethod
    def _clean_reasoning(text: str) -> str:
        cleaned = " ".join(str(text).strip().split())
        if not cleaned:
            return ""
        if re.fullmatch(r"[\[\]\(\)\s,\-0-9.eE]+", cleaned):
            return ""
        normalized = cleaned.lower()
        if normalized.startswith("output exactly") or normalized == "2 to 4 sentences":
            return ""
        cleaned = re.sub(r"^\s*(Attack|Reason|Explanation|Output)\s*:\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        if "[" in cleaned and "]" in cleaned and len(cleaned) < 120:
            return ""
        return cleaned.strip()

    def generate(self, prompt: str) -> str:
        generation_prompt = prompt if self.task == "seq2seq" else f"{prompt}\nShort reasoning:"
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
                max_new_tokens=72 if self.task == "seq2seq" else 48,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if self.prompt_echo and generated.startswith(generation_prompt):
            generated = generated[len(generation_prompt):]
        cleaned = self._clean_reasoning(generated)
        return cleaned or "High risk detected from correlated graph behavior, memory similarity, and the most important network features. Recommended action: Block."
