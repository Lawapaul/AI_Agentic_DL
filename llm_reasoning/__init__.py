"""LLM reasoning package with lazy-loaded APIs and notebook runner."""

from __future__ import annotations

from importlib import import_module

from llm_reasoning.pipelines import run_llm_reasoning

__all__ = ["LLMReasoningEngine", "LLMPipeline", "ReasoningGenerator", "run_llm_reasoning"]


def __getattr__(name: str):
    if name == "LLMReasoningEngine":
        return import_module("llm_reasoning.llm_engine").LLMReasoningEngine
    if name == "LLMPipeline":
        return import_module("llm_reasoning.llm_pipeline").LLMPipeline
    if name == "ReasoningGenerator":
        return import_module("llm_reasoning.reasoning_generator").ReasoningGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
