"""
Phase 3: Graph Correlation Layer

Builds attack-level feature profiles from Feature Gradients (FG),
then constructs an attack-class similarity graph.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Tuple

import numpy as np
try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(
            "networkx is required for Graph Correlation Layer. "
            "Install dependencies with: pip install -r requirements.txt"
        )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_attack_profiles(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fg_importance: Callable[[np.ndarray], np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Compute class-wise average FG profiles.

    Args:
        X_eval: samples with shape (N, F, 1) or (N, F)
        y_eval: class ids with shape (N,)
        fg_importance: function(sample) -> FG vector with shape (F,)

    Returns:
        dict: class_id -> mean FG profile vector
    """
    X_eval = np.asarray(X_eval)
    y_eval = np.asarray(y_eval).reshape(-1)

    if X_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("X_eval and y_eval must have same number of samples.")

    class_vectors: Dict[int, List[np.ndarray]] = {}

    for i in range(X_eval.shape[0]):
        cls = int(y_eval[i])
        sample = X_eval[i]
        imp = np.asarray(fg_importance(sample), dtype=np.float32).reshape(-1)
        class_vectors.setdefault(cls, []).append(imp)

    class_profiles: Dict[int, np.ndarray] = {}
    for cls, vectors in class_vectors.items():
        mat = np.stack(vectors, axis=0)
        class_profiles[cls] = np.mean(mat, axis=0).astype(np.float32)

    return class_profiles


def build_attack_graph(
    class_profiles: Dict[int, np.ndarray],
    threshold: float = 0.7,
) -> nx.Graph:
    """
    Build attack-class graph where edge weight is cosine similarity.
    Edge added only when similarity > threshold.
    """
    _require_networkx()
    G_attack = nx.Graph()

    for cls in class_profiles.keys():
        G_attack.add_node(int(cls))

    for a, b in combinations(class_profiles.keys(), 2):
        sim = _cosine_similarity(class_profiles[a], class_profiles[b])
        if sim > threshold:
            G_attack.add_edge(int(a), int(b), weight=float(sim))

    return G_attack


def get_top_correlated_classes(
    G_attack: nx.Graph,
    cls_id: int,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Return top correlated neighbor classes for a target class.
    """
    if cls_id not in G_attack:
        return []

    neighbors = []
    for nbr in G_attack.neighbors(cls_id):
        weight = float(G_attack[cls_id][nbr].get("weight", 0.0))
        neighbors.append((int(nbr), weight))

    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:top_k]


def get_top_correlated_pairs(
    G_attack: nx.Graph,
    top_k: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Return globally highest-weight correlated class pairs.
    """
    edges = []
    for u, v, data in G_attack.edges(data=True):
        edges.append((int(u), int(v), float(data.get("weight", 0.0))))
    edges.sort(key=lambda x: x[2], reverse=True)
    return edges[:top_k]


def validate_attack_graph(G_attack: nx.Graph) -> None:
    """
    Validation checks required by Phase 3.
    """
    if G_attack.number_of_nodes() < 1:
        raise ValueError("Attack graph must contain at least one node.")

    if any(u == v for u, v in G_attack.edges()):
        raise ValueError("Attack graph contains self-loops.")

    for _, _, data in G_attack.edges(data=True):
        if "weight" not in data:
            raise ValueError("All graph edges must have a 'weight' attribute.")


if __name__ == "__main__":
    _require_networkx()
    # Smoke test with synthetic data.
    rng = np.random.default_rng(42)
    X_test = rng.normal(size=(24, 78, 1)).astype(np.float32)
    y_test = np.array([0] * 8 + [1] * 8 + [2] * 8, dtype=np.int32)

    def _dummy_fg(sample: np.ndarray) -> np.ndarray:
        vec = np.abs(sample).reshape(-1)
        return vec.astype(np.float32)

    profiles = build_attack_profiles(X_test, y_test, _dummy_fg)
    G = build_attack_graph(profiles, threshold=0.1)
    validate_attack_graph(G)

    assert G.number_of_nodes() >= 1
    print("Graph Correlation Layer OK")
