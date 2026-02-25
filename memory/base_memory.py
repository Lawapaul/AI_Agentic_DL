"""Base interfaces and helpers for memory retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class BaseMemoryRetriever(ABC):
    """Abstract base class for retrieval-only attack memory modules."""

    def __init__(self, top_k: int = 5):
        self.top_k = int(top_k)

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={arr.shape}")
        return arr

    @staticmethod
    def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, eps)

    @staticmethod
    def _cosine_similarity(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
        q = BaseMemoryRetriever._normalize_rows(BaseMemoryRetriever._as_2d(query))
        m = BaseMemoryRetriever._normalize_rows(BaseMemoryRetriever._as_2d(memory))
        return np.dot(q, m.T)[0]

    @staticmethod
    def _to_context(labels: np.ndarray, sims: np.ndarray) -> Dict[str, object]:
        labels = np.asarray(labels).astype(int)
        sims = np.asarray(sims, dtype=np.float32)

        class_distribution: Dict[str, int] = {}
        for lab in labels.tolist():
            key = str(int(lab))
            class_distribution[key] = class_distribution.get(key, 0) + 1

        return {
            "top_k_labels": labels.tolist(),
            "similarity_scores": [float(x) for x in sims.tolist()],
            "class_distribution": class_distribution,
            "avg_similarity": float(np.mean(sims)) if sims.size > 0 else 0.0,
        }

    @abstractmethod
    def fit(
        self,
        embeddings: Optional[np.ndarray] = None,
        fg_vectors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        attack_graph=None,
    ) -> None:
        """Build memory bank for retrieval."""

    @abstractmethod
    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_fg: Optional[np.ndarray] = None,
        predicted_class: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        """Return structured memory context for a single query."""


class EmptyMemoryRetriever(BaseMemoryRetriever):
    """Baseline retriever that returns no contextual memory."""

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        return None

    def retrieve(self, query_embedding=None, query_fg=None, predicted_class=None, top_k=None):
        return {
            "top_k_labels": [],
            "similarity_scores": [],
            "class_distribution": {},
            "avg_similarity": 0.0,
        }
