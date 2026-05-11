"""Vector memory storage backend for feedback retrieval."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


class VectorMemoryFeedbackStorage:
    """Stores simple embeddings and metadata for nearest-neighbor retrieval."""

    def __init__(self, path_prefix: str):
        self.path_prefix = path_prefix
        self.vec_path = f"{path_prefix}_vectors.npy"
        self.meta_path = f"{path_prefix}_meta.jsonl"
        os.makedirs(os.path.dirname(self.vec_path), exist_ok=True)
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, object]] = []

    @staticmethod
    def _embed(features: List[float]) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros(8, dtype=np.float32)
        mean = np.mean(arr)
        std = np.std(arr)
        maxi = np.max(arr)
        mini = np.min(arr)
        med = np.median(arr)
        p75 = np.percentile(arr, 75)
        p25 = np.percentile(arr, 25)
        l2 = np.linalg.norm(arr)
        return np.array([mean, std, maxi, mini, med, p75, p25, l2], dtype=np.float32)

    def save(self, record: Dict[str, object]) -> None:
        vec = self._embed(list(record.get("features", [])))
        self.vectors.append(vec)
        self.metadata.append(record)

    def flush(self) -> None:
        if self.vectors:
            np.save(self.vec_path, np.vstack(self.vectors))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for rec in self.metadata:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    def query_recent(self, limit: int = 50) -> List[Dict[str, object]]:
        return self.metadata[-limit:]
