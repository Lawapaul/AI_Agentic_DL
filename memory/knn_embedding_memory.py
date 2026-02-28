"""Embedding-space KNN memory retrieval."""

from __future__ import annotations

from typing import Optional

import numpy as np

from memory.base_memory import BaseMemoryRetriever


class EmbeddingKNNMemory(BaseMemoryRetriever):
    """KNN memory over penultimate-layer embeddings using cosine similarity."""

    def __init__(self, top_k: int = 5):
        super().__init__(top_k=top_k)
        self.embeddings = None
        self.labels = None

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        if embeddings is None or labels is None:
            raise ValueError("EmbeddingKNNMemory requires embeddings and labels.")
        self.embeddings = self._as_2d(embeddings)
        self.labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError("Embeddings and labels must have same sample count.")

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_fg: Optional[np.ndarray] = None,
        predicted_class: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        if self.embeddings is None:
            raise RuntimeError("Memory not fitted.")
        if query_embedding is None:
            raise ValueError("Embedding query is required for EmbeddingKNNMemory.")

        k = int(top_k or self.top_k)
        sims = self._cosine_similarity(query_embedding, self.embeddings)
        top_idx = np.argsort(sims)[-k:][::-1]

        return self._to_context(self.labels[top_idx], sims[top_idx])
