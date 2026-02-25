"""Prototype (class-centroid) memory retrieval in embedding space."""

from __future__ import annotations

from typing import Optional

import numpy as np

from memory.base_memory import BaseMemoryRetriever


class PrototypeMemory(BaseMemoryRetriever):
    """Lightweight centroid memory that retrieves nearest class prototypes."""

    def __init__(self, top_k: int = 5):
        super().__init__(top_k=top_k)
        self.centroids = None
        self.centroid_labels = None

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        if embeddings is None or labels is None:
            raise ValueError("PrototypeMemory requires embeddings and labels.")

        emb = self._as_2d(embeddings)
        y = np.asarray(labels, dtype=np.int32).reshape(-1)

        classes = np.unique(y)
        centroids = []
        centroid_labels = []
        for cls in classes:
            cls_emb = emb[y == cls]
            centroids.append(np.mean(cls_emb, axis=0))
            centroid_labels.append(int(cls))

        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.centroid_labels = np.asarray(centroid_labels, dtype=np.int32)

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_fg: Optional[np.ndarray] = None,
        predicted_class: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        if self.centroids is None:
            raise RuntimeError("Memory not fitted.")
        if query_embedding is None:
            raise ValueError("Embedding query is required for PrototypeMemory.")

        k = min(int(top_k or self.top_k), self.centroids.shape[0])
        sims = self._cosine_similarity(query_embedding, self.centroids)
        top_idx = np.argsort(sims)[-k:][::-1]

        return self._to_context(self.centroid_labels[top_idx], sims[top_idx])
