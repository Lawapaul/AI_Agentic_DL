"""Structured feedback logging and reward computation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from feedback_store.feedback_db import insert_feedback


LOGGER = logging.getLogger(__name__)


def compute_reward(action_taken: str, human_action: str, severity_adjustment: float) -> float:
    """Compute a severity-scaled reward from human feedback."""

    base = 1.0 if action_taken == human_action else -1.0
    return round(base * (1.0 + abs(float(severity_adjustment))), 3)


def build_feedback_record(
    decision_output: Mapping[str, Any],
    review_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert system decision plus review into a persistent feedback record."""

    state = {
        "risk_score": float(decision_output.get("risk_score", 0.0)),
        "uncertainty": float(decision_output.get("uncertainty", 0.0)),
        "suggested_action": str(decision_output.get("suggested_action", "Monitor")),
        "explanation": str(decision_output.get("explanation", "")),
    }
    action_taken = str(decision_output.get("suggested_action", "Monitor"))
    human_action = str(review_output.get("correct_action", action_taken))
    severity_adjustment = float(review_output.get("severity_adjustment", 0.0))
    reward = compute_reward(action_taken, human_action, severity_adjustment)

    return {
        "event_id": str(decision_output.get("event_id", "unknown")),
        "state": json.dumps(state),
        "action_taken": action_taken,
        "human_action": human_action,
        "reward": reward,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def log_feedback(
    db_path: str,
    results_dir: str,
    decision_output: Mapping[str, Any],
    review_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist feedback to SQLite, JSONL, and Parquet."""

    record = build_feedback_record(decision_output, review_output)
    record_id = insert_feedback(db_path, record)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = results_path / "feedback_log.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": record_id, **record}) + "\n")

    parquet_path = results_path / "feedback_log.parquet"
    try:
        import pandas as pd

        frame = pd.DataFrame([{"id": record_id, **record}])
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - optional parquet path
        LOGGER.warning("Skipping parquet feedback export: %s", exc)

    LOGGER.info("Logged feedback for event %s with reward %.3f", record["event_id"], record["reward"])
    return {"id": record_id, **record}
