"""Feature-Gradient-space KNN memory retrieval."""

from __future__ import annotations

from typing import Optional

import numpy as np

from memory.base_memory import BaseMemoryRetriever


class FeatureGradientKNNMemory(BaseMemoryRetriever):
    """KNN memory over feature-gradient vectors using cosine similarity."""

    def __init__(self, top_k: int = 5):
        super().__init__(top_k=top_k)
        self.fg_vectors = None
        self.labels = None

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        if fg_vectors is None or labels is None:
            raise ValueError("FeatureGradientKNNMemory requires fg_vectors and labels.")
        self.fg_vectors = self._as_2d(fg_vectors)
        self.labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if self.fg_vectors.shape[0] != self.labels.shape[0]:
            raise ValueError("FG vectors and labels must have same sample count.")

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_fg: Optional[np.ndarray] = None,
        predicted_class: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        if self.fg_vectors is None:
            raise RuntimeError("Memory not fitted.")
        if query_fg is None:
            raise ValueError("FG query is required for FeatureGradientKNNMemory.")

        k = int(top_k or self.top_k)
        sims = self._cosine_similarity(query_fg, self.fg_vectors)
        top_idx = np.argsort(sims)[-k:][::-1]

        return self._to_context(self.labels[top_idx], sims[top_idx])
