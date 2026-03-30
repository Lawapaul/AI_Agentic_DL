"""Independent runner for feedback storage phase."""

from __future__ import annotations

from typing import Dict

from feedback_storage.json_storage import JSONFeedbackStorage
from feedback_storage.sqlite_storage import SQLiteFeedbackStorage
from feedback_storage.vector_memory_storage import VectorMemoryFeedbackStorage


def run_feedback_storage(record: Dict[str, object], method: str = "json", base_path: str = "experiments/results") -> None:
    """Persist feedback record using selected backend."""
    method_key = method.strip().lower()
    if method_key == "json":
        store = JSONFeedbackStorage(path=f"{base_path}/feedback.jsonl")
        store.save(record)
        return
    if method_key == "sqlite":
        store = SQLiteFeedbackStorage(db_path=f"{base_path}/feedback.db")
        store.save(record)
        return
    if method_key == "vector":
        store = VectorMemoryFeedbackStorage(path_prefix=f"{base_path}/feedback_vector")
        store.save(record)
        store.flush()
        return
    raise ValueError(f"Unknown feedback storage method: {method}")
