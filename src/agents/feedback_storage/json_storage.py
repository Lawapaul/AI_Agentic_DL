"""JSON line-based feedback storage backend."""

from __future__ import annotations

import json
import os
from typing import Dict, List


class JSONFeedbackStorage:
    """Stores each feedback record as one JSON line."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, record: Dict[str, object]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def query_recent(self, limit: int = 50) -> List[Dict[str, object]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line) for line in lines]
