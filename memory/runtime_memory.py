"""Runtime-growing memory for demo-safe retrieval."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from memory.base_memory import BaseMemoryRetriever


class RuntimeMemoryRetriever(BaseMemoryRetriever):
    """Append-only memory that retrieves similar prior samples during one run."""

    def __init__(self, top_k: int = 1, min_similarity: float = 0.5):
        super().__init__(top_k=top_k)
        self.min_similarity = float(min_similarity)
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int32)

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        del attack_graph
        bank = embeddings if embeddings is not None else fg_vectors
        if bank is None or labels is None:
            self.vectors = np.empty((0, 0), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int32)
            return
        self.vectors = self._as_2d(bank)
        self.labels = np.asarray(labels, dtype=np.int32).reshape(-1)

    def add(self, vector: np.ndarray, label: int) -> None:
        row = self._as_2d(vector)
        if self.vectors.size == 0:
            self.vectors = row
        else:
            self.vectors = np.vstack([self.vectors, row])
        self.labels = np.concatenate([self.labels, np.asarray([int(label)], dtype=np.int32)])

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_fg: Optional[np.ndarray] = None,
        predicted_class: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        query = query_embedding if query_embedding is not None else query_fg
        if query is None:
            raise ValueError("RuntimeMemoryRetriever requires query_embedding or query_fg.")

        if self.vectors.size == 0 or self.labels.size == 0:
            fallback_label = 0 if predicted_class is None else int(predicted_class)
            return self._to_context(np.asarray([fallback_label], dtype=np.int32), np.asarray([0.5], dtype=np.float32))

        sims = self._cosine_similarity(query, self.vectors)
        k = min(int(top_k or self.top_k), len(sims))
        top_idx = np.argsort(sims)[-k:][::-1]
        top_sims = sims[top_idx]
        top_labels = self.labels[top_idx]

        if float(top_sims[0]) < self.min_similarity:
            fallback_label = int(top_labels[0])
            fallback_sim = max(float(top_sims[0]), self.min_similarity)
            return self._to_context(np.asarray([fallback_label], dtype=np.int32), np.asarray([fallback_sim], dtype=np.float32))

        return self._to_context(top_labels, top_sims)
