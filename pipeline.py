"""Legacy compatibility entry point for the reorganized project."""

from src.agents.orchestrator import run_pipeline, run_showcase_demo

__all__ = ["run_pipeline", "run_showcase_demo"]


if __name__ == "__main__":
    run_pipeline(mode="auto")
