"""Adaptive risk fusion package with lazy export of the fusion factory."""

from __future__ import annotations

from importlib import import_module

__all__ = ["FusionFactory"]


def __getattr__(name: str):
    if name != "FusionFactory":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module("src.fusion.risk_fusion.fusion_factory").FusionFactory
