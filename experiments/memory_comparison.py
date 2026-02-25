"""Phase 4: Memory Retrieval strategy comparison on full CICIDS training."""

from __future__ import annotations

import argparse
import os
import time
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

from data.loader import IDSDataLoader
from explainability.feature_gradient_explainer import create_feature_gradient_explainer
from graph_correlation import build_attack_graph, build_attack_profiles
from memory import (
    EmbeddingKNNMemory,
    EmptyMemoryRetriever,
    FeatureGradientKNNMemory,
    GraphAwareMemory,
    MemoryEvaluator,
    PrototypeMemory,
)
from models.trainer import IDSModelTrainer


def compute_fg_matrix(explainer, X: np.ndarray, log_every: int = 1000) -> np.ndarray:
    """Compute FG vectors for all samples in X."""
    fg_vectors: List[np.ndarray] = []
    total = X.shape[0]
    for i in range(total):
        fg_vectors.append(explainer.feature_importance(X[i]))
        if (i + 1) % log_every == 0:
            print(f"Computed FG vectors: {i + 1}/{total}")
    return np.asarray(fg_vectors, dtype=np.float32)


def evaluate_strategy(
    strategy_name: str,
    memory,
    evaluator: MemoryEvaluator,
    test_embeddings: np.ndarray,
    test_fg: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    """Run retrieval on test set and compute all Phase 4 metrics."""
    contexts = []
    for i in range(test_embeddings.shape[0]):
        contexts.append(
            memory.retrieve(
                query_embedding=test_embeddings[i],
                query_fg=test_fg[i] if test_fg is not None else None,
                predicted_class=int(y_pred[i]),
                top_k=top_k,
            )
        )

    metrics = evaluator.evaluate_contexts(contexts, y_test)

    if isinstance(memory, EmptyMemoryRetriever):
        stability = 0.0
    else:
        stability = evaluator.retrieval_stability(
            memory=memory,
            query_embeddings=test_embeddings,
            query_fgs=test_fg,
            predicted_classes=y_pred,
        )

    row = {
        "strategy": strategy_name,
        "top1_retrieval_accuracy": metrics["top1_retrieval_accuracy"],
        "top5_retrieval_accuracy": metrics["top5_retrieval_accuracy"],
        "class_purity": metrics["class_purity"],
        "retrieval_entropy": metrics["retrieval_entropy"],
        "avg_cosine_similarity": metrics["avg_cosine_similarity"],
        "retrieval_stability": stability,
    }
    return row


def run_memory_comparison(
    model_path: str = "saved_models/hybrid_memory_phase4/hybrid_latest.keras",
    output_csv: str = "memory_comparison_results.csv",
    epochs: int = 10,
    top_k: int = 5,
    force_retrain: bool = False,
):
    """Train (full dataset), build memory banks, and compare Phase 4 strategies."""
    print("=" * 72)
    print("PHASE 4 MEMORY COMPARISON (FULL DATASET)")
    print("=" * 72)

    device = IDSModelTrainer.detect_device()
    batch_size = IDSModelTrainer.recommended_batch_size()
    print(f"Device detected: {device}")
    print(f"Batch size selected: {batch_size}")

    loader = IDSDataLoader(balanced_total_samples=None)
    data = loader.load_and_preprocess()

    trainer = IDSModelTrainer(model_type="hybrid", model_save_path=model_path)
    if os.path.exists(model_path) and not force_retrain:
        print(f"Using cached trained model: {model_path}")
        trainer.model = IDSModelTrainer.load_model(model_path)
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        start = time.time()
        trainer.train(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            epochs=epochs,
            batch_size=batch_size,
        )
        trainer.save_model(model_path)
        versioned_model_path = os.path.join(
            os.path.dirname(model_path),
            f"hybrid_epoch{epochs}_{time.strftime('%Y%m%d_%H%M%S')}.keras",
        )
        shutil.copy2(model_path, versioned_model_path)
        print(f"Versioned model snapshot saved to: {versioned_model_path}")
        print(f"Training wall time (sec): {time.time() - start:.2f}")

    explainer = create_feature_gradient_explainer(trainer.model, data["feature_names"])

    print("\nExtracting embeddings...")
    train_embeddings = trainer.extract_embeddings(data["X_train"], batch_size=batch_size)
    test_embeddings = trainer.extract_embeddings(data["X_test"], batch_size=batch_size)

    print("\nComputing FG vectors for train/test...")
    train_fg = compute_fg_matrix(explainer, data["X_train"])
    test_fg = compute_fg_matrix(explainer, data["X_test"])

    print("\nBuilding Phase 3 graph for Graph-Aware Memory...")
    class_profiles = build_attack_profiles(
        X_eval=data["X_train"],
        y_eval=data["y_train"],
        fg_importance=explainer.feature_importance,
    )
    attack_graph = build_attack_graph(class_profiles=class_profiles, threshold=0.7)

    y_pred = trainer.predict(data["X_test"], return_probabilities=False)

    memories = {}

    no_memory = EmptyMemoryRetriever(top_k=top_k)
    no_memory.fit()
    memories["No Memory (baseline)"] = no_memory

    emb_mem = EmbeddingKNNMemory(top_k=top_k)
    emb_mem.fit(embeddings=train_embeddings, labels=data["y_train"])
    memories["Embedding KNN"] = emb_mem

    fg_mem = FeatureGradientKNNMemory(top_k=top_k)
    fg_mem.fit(fg_vectors=train_fg, labels=data["y_train"])
    memories["FG KNN"] = fg_mem

    proto_mem = PrototypeMemory(top_k=top_k)
    proto_mem.fit(embeddings=train_embeddings, labels=data["y_train"])
    memories["Prototype Memory"] = proto_mem

    graph_mem = GraphAwareMemory(top_k=top_k)
    graph_mem.fit(
        embeddings=train_embeddings,
        labels=data["y_train"],
        attack_graph=attack_graph,
    )
    memories["Graph-Aware Memory"] = graph_mem

    evaluator = MemoryEvaluator(top_k=top_k)

    rows = []
    for name, memory in memories.items():
        print(f"\nEvaluating: {name}")
        rows.append(
            evaluate_strategy(
                strategy_name=name,
                memory=memory,
                evaluator=evaluator,
                test_embeddings=test_embeddings,
                test_fg=test_fg,
                y_test=data["y_test"],
                y_pred=y_pred,
                top_k=top_k,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    pretty = df.copy()
    metric_cols = [c for c in pretty.columns if c != "strategy"]
    for col in metric_cols:
        pretty[col] = pretty[col].map(lambda v: f"{float(v):.4f}")

    print("\n" + "=" * 72)
    print("MEMORY RETRIEVAL COMPARISON TABLE")
    print("=" * 72)
    print(pretty.to_string(index=False))
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 memory retrieval comparison")
    parser.add_argument("--model-path", type=str, default="saved_models/hybrid_memory_phase4/hybrid_latest.keras")
    parser.add_argument("--output", type=str, default="memory_comparison_results.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    run_memory_comparison(
        model_path=args.model_path,
        output_csv=args.output,
        epochs=args.epochs,
        top_k=args.top_k,
        force_retrain=args.retrain,
    )
