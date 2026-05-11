"""CLI entry point for the reorganized AI Agentic IDS repository."""

from __future__ import annotations

import argparse
import json

from src.agents.orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options for full or demo pipeline execution."""

    parser = argparse.ArgumentParser(description="Run the AI Agentic IDS pipeline.")
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "demo"],
        default="auto",
        help="Choose 'full' to require processed data and a trained model, or 'demo' for the lightweight showcase flow.",
    )
    parser.add_argument("--processed-path", default=None, help="Optional processed dataset directory for full pipeline execution.")
    parser.add_argument("--model-path", default=None, help="Optional trained model path for full pipeline execution.")
    parser.add_argument("--config", default=None, help="Optional config file path.")
    return parser.parse_args()


def main() -> dict:
    """Run the selected pipeline mode and print a compact JSON summary."""

    args = parse_args()
    summary = run_pipeline(
        mode=args.mode,
        processed_path=args.processed_path,
        model_path=args.model_path,
        config_path=args.config,
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
