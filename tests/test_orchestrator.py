"""Basic smoke tests for the GitHub-ready demo entry point."""

from src.agents.orchestrator import run_showcase_demo


def test_showcase_demo_returns_records():
    summary = run_showcase_demo()
    assert summary["status"] == "completed"
    assert summary["mode"] == "demo"
    assert len(summary["records"]) > 0
