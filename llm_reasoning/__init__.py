"""Lightweight Transformer-based reasoning package for IDS explanations."""

from __future__ import annotations

from importlib import import_module

__all__ = ["LLMReasoningEngine", "LLMPipeline", "ReasoningGenerator"]


def __getattr__(name: str):
    if name == "LLMReasoningEngine":
        return import_module("llm_reasoning.llm_engine").LLMReasoningEngine
    if name == "LLMPipeline":
        return import_module("llm_reasoning.llm_pipeline").LLMPipeline
    if name == "ReasoningGenerator":
        return import_module("llm_reasoning.reasoning_generator").ReasoningGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
