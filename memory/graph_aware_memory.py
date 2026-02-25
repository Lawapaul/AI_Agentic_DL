"""Graph-aware memory retrieval using Phase 3 correlation graph."""

from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np

from memory.base_memory import BaseMemoryRetriever


class GraphAwareMemory(BaseMemoryRetriever):
    """Retrieval from same + graph-neighbor classes, weighted by edge similarity."""

    def __init__(self, top_k: int = 5):
        super().__init__(top_k=top_k)
        self.embeddings = None
        self.labels = None
        self.attack_graph = None

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        if embeddings is None or labels is None:
            raise ValueError("GraphAwareMemory requires embeddings and labels.")
        self.embeddings = self._as_2d(embeddings)
        self.labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        self.attack_graph = attack_graph

    def _class_weights(self, predicted_class: int) -> Dict[int, float]:
        weights: Dict[int, float] = {int(predicted_class): 1.0}
        if self.attack_graph is None or predicted_class not in self.attack_graph:
            return weights

        for nbr in self.attack_graph.neighbors(int(predicted_class)):
            w = float(self.attack_graph[int(predicted_class)][int(nbr)].get("weight", 0.0))
            if w > 0.0:
                weights[int(nbr)] = max(weights.get(int(nbr), 0.0), w)
        return weights

    def _candidate_mask(self, allowed_classes: Set[int]) -> np.ndarray:
        mask = np.zeros(self.labels.shape[0], dtype=bool)
        for cls in allowed_classes:
            mask |= self.labels == int(cls)
        return mask

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
            raise ValueError("Embedding query is required for GraphAwareMemory.")
        if predicted_class is None:
            raise ValueError("predicted_class is required for GraphAwareMemory.")

        k = int(top_k or self.top_k)
        class_weights = self._class_weights(int(predicted_class))

        mask = self._candidate_mask(set(class_weights.keys()))
        if not np.any(mask):
            mask = np.ones(self.labels.shape[0], dtype=bool)

        candidate_emb = self.embeddings[mask]
        candidate_labels = self.labels[mask]

        sims = self._cosine_similarity(query_embedding, candidate_emb)
        weighted_sims = np.asarray(
            [float(s) * float(class_weights.get(int(lbl), 1.0)) for s, lbl in zip(sims, candidate_labels)],
            dtype=np.float32,
        )

        top_idx = np.argsort(weighted_sims)[-k:][::-1]
        return self._to_context(candidate_labels[top_idx], weighted_sims[top_idx])
