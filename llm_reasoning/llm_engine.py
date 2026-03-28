"""Prompt construction for lightweight LLM-based IDS reasoning."""

from __future__ import annotations

from typing import Iterable


class LLMReasoningEngine:
    """Build concise prompts from model outputs and feature vectors."""

    def __init__(self, feature_count: int = 10):
        self.feature_count = feature_count

    def generate_prompt(self, features: Iterable[float], prediction, confidence: float) -> str:
        feature_list = list(features)
        prompt = f"""
You are a cybersecurity expert analyzing network traffic.

Feature Summary (first {self.feature_count} values): {feature_list[: self.feature_count]}
Predicted Class: {prediction}
Confidence Score: {confidence:.2f}

Based on this, answer clearly:
1. What type of attack is this?
2. Why does this pattern indicate that attack?
3. What is the severity (Low/Medium/High)?
4. What action should be taken?

Provide a concise explanation with short labeled sections.
"""
        return prompt.strip()
