"""Evaluation utilities for Phase 4 memory retrieval."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_entropy(labels: List[int]) -> float:
    if not labels:
        return 0.0
    arr = np.asarray(labels, dtype=np.int32)
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))


def _class_purity(labels: List[int], true_label: int) -> float:
    if not labels:
        return 0.0
    arr = np.asarray(labels, dtype=np.int32)
    return float(np.mean(arr == int(true_label)))


class MemoryEvaluator:
    """Computes retrieval metrics across memory strategies."""

    def __init__(self, top_k: int = 5):
        self.top_k = int(top_k)

    def evaluate_contexts(self, contexts: List[Dict[str, object]], y_true: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)

        top1_hits = []
        top5_hits = []
        purities = []
        entropies = []
        avg_sims = []

        for i, ctx in enumerate(contexts):
            labels = [int(x) for x in ctx.get("top_k_labels", [])]
            sims = [float(x) for x in ctx.get("similarity_scores", [])]
            yt = int(y_true[i])

            top1_hits.append(1.0 if len(labels) > 0 and labels[0] == yt else 0.0)
            top5_hits.append(1.0 if yt in labels[:5] else 0.0)
            purities.append(_class_purity(labels, yt))
            entropies.append(_safe_entropy(labels))
            avg_sims.append(float(np.mean(sims)) if sims else 0.0)

        return {
            "top1_retrieval_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
            "top5_retrieval_accuracy": float(np.mean(top5_hits)) if top5_hits else 0.0,
            "class_purity": float(np.mean(purities)) if purities else 0.0,
            "retrieval_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "avg_cosine_similarity": float(np.mean(avg_sims)) if avg_sims else 0.0,
        }

    def retrieval_stability(
        self,
        memory,
        query_embeddings: np.ndarray,
        query_fgs: Optional[np.ndarray] = None,
        predicted_classes: Optional[np.ndarray] = None,
        noise_std: float = 1e-3,
    ) -> float:
        rng = np.random.default_rng(42)
        q_emb = np.asarray(query_embeddings, dtype=np.float32)
        q_fgs = None if query_fgs is None else np.asarray(query_fgs, dtype=np.float32)

        if predicted_classes is None:
            predicted_classes = np.zeros((q_emb.shape[0],), dtype=np.int32)

        overlaps = []
        for i in range(q_emb.shape[0]):
            emb = q_emb[i]
            fg = None if q_fgs is None else q_fgs[i]
            pred = int(predicted_classes[i])

            c1 = memory.retrieve(query_embedding=emb, query_fg=fg, predicted_class=pred, top_k=self.top_k)
            perturbed = emb + rng.normal(0.0, noise_std, size=emb.shape).astype(np.float32)
            c2 = memory.retrieve(query_embedding=perturbed, query_fg=fg, predicted_class=pred, top_k=self.top_k)

            s1 = set(int(x) for x in c1.get("top_k_labels", []))
            s2 = set(int(x) for x in c2.get("top_k_labels", []))
            denom = max(len(s1 | s2), 1)
            overlaps.append(len(s1 & s2) / denom)

        return float(np.mean(overlaps)) if overlaps else 0.0

    @staticmethod
    def to_dataframe(rows: List[Dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(rows)
