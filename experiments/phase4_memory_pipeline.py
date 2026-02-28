"""Standalone Phase 4 memory retrieval experiment pipeline.

This script is isolated for Phase 4 testing and does not modify the main
pipeline or later phases.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from explainability.feature_gradient_explainer import create_feature_gradient_explainer
from graph_correlation import build_attack_graph, build_attack_profiles, validate_attack_graph
from memory import (
    EmbeddingKNNMemory,
    EmptyMemoryRetriever,
    FeatureGradientKNNMemory,
    GraphAwareMemory,
    MemoryEvaluator,
    PrototypeMemory,
)
from memory.base_memory import BaseMemoryRetriever
from models.trainer import IDSModelTrainer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


MODEL_PATH_DEFAULT = "saved_models/phase4_memory/hybrid_latest.keras"
RESULT_CSV_DEFAULT = "phase4_memory_results.csv"
GRAPH_THRESHOLD_DEFAULT = 0.7
RNG_SEED = 42
MAX_FG_TRAIN_SAMPLES = 10000


@dataclass
class PipelineArtifacts:
    data: Dict[str, object]
    trainer: IDSModelTrainer
    train_embeddings: np.ndarray
    test_embeddings: np.ndarray
    fg_bank_embeddings: np.ndarray
    fg_bank_labels: np.ndarray
    train_fg: np.ndarray
    test_fg: np.ndarray
    y_pred_test: np.ndarray
    attack_graph: object


class CombinedMemory(BaseMemoryRetriever):
    """KNN memory that combines embedding and FG cosine similarities."""

    def __init__(self, top_k: int = 5, embedding_weight: float = 0.5):
        super().__init__(top_k=top_k)
        if not (0.0 <= embedding_weight <= 1.0):
            raise ValueError("embedding_weight must be in [0, 1].")
        self.embedding_weight = float(embedding_weight)
        self.fg_weight = 1.0 - self.embedding_weight
        self.embeddings: Optional[np.ndarray] = None
        self.fg_vectors: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def fit(self, embeddings=None, fg_vectors=None, labels=None, attack_graph=None) -> None:
        del attack_graph
        if embeddings is None or fg_vectors is None or labels is None:
            raise ValueError("CombinedMemory requires embeddings, fg_vectors, and labels.")

        self.embeddings = self._as_2d(embeddings)
        self.fg_vectors = self._as_2d(fg_vectors)
        self.labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        n = self.labels.shape[0]
        if self.embeddings.shape[0] != n or self.fg_vectors.shape[0] != n:
            raise ValueError("Embeddings/FG/labels must share the same number of samples.")

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        query_fg: Optional[np.ndarray] = None,
        predicted_class: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        del predicted_class
        if self.embeddings is None or self.fg_vectors is None or self.labels is None:
            raise RuntimeError("Memory not fitted.")
        if query_embedding is None or query_fg is None:
            raise ValueError("CombinedMemory requires both query_embedding and query_fg.")

        k = int(top_k or self.top_k)
        emb_sims = self._cosine_similarity(query_embedding, self.embeddings)
        fg_sims = self._cosine_similarity(query_fg, self.fg_vectors)

        combined = (self.embedding_weight * emb_sims) + (self.fg_weight * fg_sims)
        top_idx = np.argsort(combined)[-k:][::-1]
        return self._to_context(self.labels[top_idx], combined[top_idx])


def get_branch_name() -> str:
    try:
        out = subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip() or "unknown"
    except Exception:
        return "unknown"


def print_device_info() -> str:
    device = IDSModelTrainer.detect_device()
    print(f"Device detected: {device}")
    if device == "GPU":
        gpus = tf.config.list_physical_devices("GPU")
        names = [gpu.name for gpu in gpus]
        print(f"GPU devices: {names}")
    else:
        print("GPU devices: []")
    return device


def validate_embeddings(train_embeddings: np.ndarray, test_embeddings: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D. Got train={train_embeddings.shape}, test={test_embeddings.shape}")
    if train_embeddings.shape[0] != y_train.shape[0]:
        raise ValueError("Train embeddings count does not match y_train.")
    if test_embeddings.shape[0] != y_test.shape[0]:
        raise ValueError("Test embeddings count does not match y_test.")
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError("Train/Test embedding dimensions do not match.")


def validate_fg_vectors(train_fg: np.ndarray, test_fg: np.ndarray) -> None:
    if train_fg.ndim != 2 or test_fg.ndim != 2:
        raise ValueError(f"FG vectors must be 2D. Got train={train_fg.shape}, test={test_fg.shape}")
    if train_fg.shape[1] != test_fg.shape[1]:
        raise ValueError("Train/Test FG dimensions do not match.")

    train_norm = np.linalg.norm(train_fg, axis=1)
    test_norm = np.linalg.norm(test_fg, axis=1)
    if np.allclose(train_norm, 0.0):
        raise ValueError("All train FG vectors are zero.")
    if np.allclose(test_norm, 0.0):
        raise ValueError("All test FG vectors are zero.")


def _ensure_feature_matrix(X: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (N, F) or 3D (N, F, 1). Got shape={arr.shape}")
    return arr


def load_processed_data(processed_path: str) -> Dict[str, object]:
    X_train = np.load(os.path.join(processed_path, "X_train.npy"))
    X_test = np.load(os.path.join(processed_path, "X_test.npy"))
    y_train = np.load(os.path.join(processed_path, "y_train.npy"))
    y_test = np.load(os.path.join(processed_path, "y_test.npy"))

    X_train = _ensure_feature_matrix(X_train, "X_train")
    X_test = _ensure_feature_matrix(X_test, "X_test")
    y_train = np.asarray(y_train, dtype=np.int32).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.int32).reshape(-1)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train sample counts do not match.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test sample counts do not match.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test feature counts do not match.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": None,  # Only needed if explainer requires
    }


def compute_fg_matrix(explainer, X: np.ndarray, desc: str) -> np.ndarray:
    fg_vectors: List[np.ndarray] = []
    for i in tqdm(range(X.shape[0]), desc=desc):
        fg_vectors.append(explainer.feature_importance(X[i]))
    return np.asarray(fg_vectors, dtype=np.float32)


def sample_fg_subset(X_train: np.ndarray, y_train: np.ndarray, train_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X_train.shape[0] <= MAX_FG_TRAIN_SAMPLES:
        print(f"FG subset: using full training set ({X_train.shape[0]} samples)")
        return X_train, y_train, train_embeddings

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(X_train.shape[0], MAX_FG_TRAIN_SAMPLES, replace=False)
    print(f"FG subset: using random {MAX_FG_TRAIN_SAMPLES} / {X_train.shape[0]} training samples")
    return X_train[idx], y_train[idx], train_embeddings[idx]


def build_graph_with_safety(X_train: np.ndarray, y_train: np.ndarray, explainer, start_threshold: float = GRAPH_THRESHOLD_DEFAULT):
    thresholds = [start_threshold, 0.6, 0.5, 0.4, 0.3, 0.2]
    class_profiles = build_attack_profiles(
        X_eval=X_train,
        y_eval=y_train,
        fg_importance=explainer.feature_importance,
    )

    graph = None
    selected = None
    for threshold in thresholds:
        graph = build_attack_graph(class_profiles=class_profiles, threshold=threshold)
        if graph.number_of_edges() > 0:
            selected = threshold
            break

    if graph is None:
        raise RuntimeError("Failed to build attack graph.")
    if graph.number_of_edges() == 0:
        raise ValueError("Attack graph contains zero edges across tested thresholds.")

    validate_attack_graph(graph)
    print(f"Graph built with threshold={selected:.2f}, nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    return graph


def resolve_model_path(model_path: str) -> str:
    """Resolve model target path for both file and directory inputs."""
    path = os.path.expanduser(model_path)
    has_file_ext = path.endswith(".keras") or path.endswith(".h5")
    if has_file_ext:
        model_dir = os.path.dirname(path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        return path

    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "hybrid_latest.keras")


def train_or_load_hybrid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    retrain: bool,
    model_path: str,
) -> IDSModelTrainer:
    resolved_model_path = resolve_model_path(model_path)
    trainer = IDSModelTrainer(model_type="hybrid", model_save_path=resolved_model_path)

    model_exists = os.path.exists(resolved_model_path)
    print(f"Model path resolved to: {resolved_model_path}")
    print(f"Saved model exists: {model_exists}")

    if model_exists and not retrain:
        print(f"Loading existing model: {resolved_model_path}")
        trainer.model = IDSModelTrainer.load_model(resolved_model_path)
    else:
        train_batch_size = IDSModelTrainer.recommended_batch_size()
        if retrain and model_exists:
            print("Retrain requested: training a fresh model and overwriting saved model.")
        else:
            print("Saved model not found: training new model.")
        print(f"Training Hybrid CNN-LSTM on full dataset with batch_size={train_batch_size}")
        trainer.train(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=10,
            batch_size=train_batch_size,
        )
        trainer.save_model(resolved_model_path)

    return trainer


def prepare_artifacts(
    retrain: bool,
    embedding_batch_size: int,
    model_path: str,
    processed_path: str,
) -> PipelineArtifacts:
    print(f"Loading processed data from: {processed_path}")
    data = load_processed_data(processed_path)

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.int32)
    y_test = np.asarray(data["y_test"], dtype=np.int32)
    if data.get("feature_names") is None:
        data["feature_names"] = [f"feature_{i}" for i in range(X_train.shape[1])]

    trainer = train_or_load_hybrid(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        retrain=retrain,
        model_path=model_path,
    )

    print("Extracting embeddings (penultimate layer)...")
    train_embeddings = trainer.extract_embeddings(X_train, batch_size=embedding_batch_size)
    test_embeddings = trainer.extract_embeddings(X_test, batch_size=embedding_batch_size)
    validate_embeddings(train_embeddings, test_embeddings, y_train, y_test)

    X_train_fg, y_train_fg, train_embeddings_fg = sample_fg_subset(X_train, y_train, train_embeddings)

    print("Generating Feature Gradient vectors...")
    explainer = create_feature_gradient_explainer(trainer.model, data["feature_names"])
    train_fg = compute_fg_matrix(explainer, X_train_fg, "FG train")
    test_fg = compute_fg_matrix(explainer, X_test, "FG test")
    validate_fg_vectors(train_fg, test_fg)

    print("Building graph correlation...")
    attack_graph = build_graph_with_safety(X_train_fg, y_train_fg, explainer)

    y_pred_test = trainer.predict(X_test, return_probabilities=False).astype(np.int32)
    if y_pred_test.shape[0] != y_test.shape[0]:
        raise ValueError("Prediction count mismatch on test set.")

    return PipelineArtifacts(
        data=data,
        trainer=trainer,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        fg_bank_embeddings=train_embeddings_fg,
        fg_bank_labels=y_train_fg,
        train_fg=train_fg,
        test_fg=test_fg,
        y_pred_test=y_pred_test,
        attack_graph=attack_graph,
    )


def build_strategies(top_k: int) -> Dict[str, BaseMemoryRetriever]:
    return {
        "NoMemory": EmptyMemoryRetriever(top_k=top_k),
        "EmbeddingKNN": EmbeddingKNNMemory(top_k=top_k),
        "FGKNN": FeatureGradientKNNMemory(top_k=top_k),
        "PrototypeMemory": PrototypeMemory(top_k=top_k),
        "GraphAwareMemory": GraphAwareMemory(top_k=top_k),
        "CombinedMemory": CombinedMemory(top_k=top_k, embedding_weight=0.5),
    }


def fit_strategy(
    name: str,
    memory: BaseMemoryRetriever,
    train_embeddings: np.ndarray,
    fg_bank_embeddings: np.ndarray,
    fg_bank_labels: np.ndarray,
    train_fg: np.ndarray,
    y_train: np.ndarray,
    attack_graph: object,
) -> int:
    if name == "NoMemory":
        memory.fit()
        return 0
    if name == "EmbeddingKNN":
        memory.fit(embeddings=train_embeddings, labels=y_train)
        return int(train_embeddings.shape[0])
    if name == "FGKNN":
        memory.fit(fg_vectors=train_fg, labels=fg_bank_labels)
        return int(train_fg.shape[0])
    if name == "PrototypeMemory":
        memory.fit(embeddings=train_embeddings, labels=y_train)
        return int(len(np.unique(y_train)))
    if name == "GraphAwareMemory":
        memory.fit(embeddings=train_embeddings, labels=y_train, attack_graph=attack_graph)
        return int(train_embeddings.shape[0])
    if name == "CombinedMemory":
        memory.fit(embeddings=fg_bank_embeddings, fg_vectors=train_fg, labels=fg_bank_labels)
        return int(train_fg.shape[0])
    raise ValueError(f"Unknown memory strategy: {name}")


def evaluate_strategy(
    strategy_name: str,
    memory: BaseMemoryRetriever,
    evaluator: MemoryEvaluator,
    test_embeddings: np.ndarray,
    test_fg: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    top_k: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    contexts: List[Dict[str, object]] = []
    retrieval_times: List[float] = []

    for i in tqdm(range(test_embeddings.shape[0]), desc=f"Eval {strategy_name}"):
        start = time.perf_counter()
        ctx = memory.retrieve(
            query_embedding=test_embeddings[i],
            query_fg=test_fg[i],
            predicted_class=int(y_pred_test[i]),
            top_k=top_k,
        )
        retrieval_times.append(time.perf_counter() - start)
        contexts.append(ctx)

    metrics = evaluator.evaluate_contexts(contexts, y_test)
    metrics["retrieval_time_sec"] = float(np.mean(retrieval_times)) if retrieval_times else 0.0
    metrics["strategy"] = strategy_name
    example = contexts[0] if contexts else {}
    return metrics, example


def run_pipeline(args: argparse.Namespace) -> None:
    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)

    print("=" * 53)
    print("PHASE 4 MEMORY RETRIEVAL EXPERIMENT PIPELINE")
    print("=" * 53)
    print(f"Branch name: {get_branch_name()}")
    print(f"File created: experiments/phase4_memory_pipeline.py")
    device = print_device_info()

    artifacts = prepare_artifacts(
        retrain=bool(args.retrain),
        embedding_batch_size=int(args.batch_size),
        model_path=args.model_path,
        processed_path=args.processed_path,
    )

    y_train = np.asarray(artifacts.data["y_train"], dtype=np.int32)
    y_test = np.asarray(artifacts.data["y_test"], dtype=np.int32)
    memory_bank_size = int(y_train.shape[0])

    if memory_bank_size <= 0:
        raise ValueError("Memory bank size must be > 0.")

    print(f"Number of training samples: {y_train.shape[0]}")
    print(f"Number of test samples: {y_test.shape[0]}")
    print(f"Memory bank size: {memory_bank_size}")

    strategies = build_strategies(top_k=int(args.top_k))

    if args.memory:
        selected = args.memory.strip()
        if selected not in strategies:
            raise ValueError(
                f"Unknown --memory '{selected}'. Valid: {', '.join(strategies.keys())}"
            )
        strategy_names = [selected]
    else:
        strategy_names = list(strategies.keys()) if args.compare_all else list(strategies.keys())

    evaluator = MemoryEvaluator(top_k=int(args.top_k))
    rows: List[Dict[str, float]] = []
    example_outputs: Dict[str, Dict[str, object]] = {}

    for name in strategy_names:
        memory = strategies[name]
        fitted_size = fit_strategy(
            name=name,
            memory=memory,
            train_embeddings=artifacts.train_embeddings,
            fg_bank_embeddings=artifacts.fg_bank_embeddings,
            fg_bank_labels=artifacts.fg_bank_labels,
            train_fg=artifacts.train_fg,
            y_train=y_train,
            attack_graph=artifacts.attack_graph,
        )
        if name != "NoMemory" and fitted_size <= 0:
            raise ValueError(f"{name} memory bank size must be > 0.")

        row, sample_ctx = evaluate_strategy(
            strategy_name=name,
            memory=memory,
            evaluator=evaluator,
            test_embeddings=artifacts.test_embeddings,
            test_fg=artifacts.test_fg,
            y_test=y_test,
            y_pred_test=artifacts.y_pred_test,
            top_k=int(args.top_k),
        )
        row["memory_bank_size"] = float(fitted_size)
        rows.append(row)
        example_outputs[name] = sample_ctx

    results_df = pd.DataFrame(rows)[
        [
            "strategy",
            "top1_retrieval_accuracy",
            "top5_retrieval_accuracy",
            "class_purity",
            "retrieval_entropy",
            "avg_cosine_similarity",
            "retrieval_time_sec",
            "memory_bank_size",
        ]
    ]
    results_df.to_csv(args.output, index=False)

    best_idx = results_df["top1_retrieval_accuracy"].idxmax()
    best_strategy = str(results_df.loc[best_idx, "strategy"])

    print("\n" + "=" * 53)
    print("PHASE 4 MEMORY COMPARISON RESULTS")
    print("=" * 53)
    header = f"{'Strategy':<18} {'Top-1':>7} {'Top-5':>7} {'Purity':>8} {'Entropy':>8} {'Cosine':>8} {'Time(s)':>9}"
    print(header)
    print("-" * len(header))
    for _, row in results_df.iterrows():
        print(
            f"{str(row['strategy']):<18} "
            f"{float(row['top1_retrieval_accuracy']):>7.4f} "
            f"{float(row['top5_retrieval_accuracy']):>7.4f} "
            f"{float(row['class_purity']):>8.4f} "
            f"{float(row['retrieval_entropy']):>8.4f} "
            f"{float(row['avg_cosine_similarity']):>8.4f} "
            f"{float(row['retrieval_time_sec']):>9.6f}"
        )
    print("-" * len(header))
    print(f"BEST MEMORY STRATEGY: {best_strategy}")
    print("=" * 53)

    example = example_outputs.get(best_strategy) or next(iter(example_outputs.values()), {})
    print(f"Example retrieval output ({best_strategy}, first test sample): {example}")
    print(f"GPU detected or CPU: {device}")
    print(f"Results saved to: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 memory retrieval strategy pipeline")
    parser.add_argument(
        "--processed_path",
        type=str,
        required=True,
        help="Path to processed .npy files",
    )
    parser.add_argument("--retrain", action="store_true", help="Retrain model on full dataset")
    parser.add_argument("--compare_all", action="store_true", help="Run all memory strategies")
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        choices=["NoMemory", "EmbeddingKNN", "FGKNN", "PrototypeMemory", "GraphAwareMemory", "CombinedMemory"],
        help="Run a single memory strategy",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval size")
    parser.add_argument("--batch_size", type=int, default=512, help="Embedding extraction batch size")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH_DEFAULT, help="Hybrid model path")
    parser.add_argument("--output", type=str, default=RESULT_CSV_DEFAULT, help="Results CSV output path")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.memory is None and not args.compare_all:
        args.compare_all = True
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
