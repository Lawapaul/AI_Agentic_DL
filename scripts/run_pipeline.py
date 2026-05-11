"""Convenience script for auto-selecting full or demo pipeline execution."""

from src.agents.orchestrator import run_pipeline


if __name__ == "__main__":
    run_pipeline(mode="auto")
