"""
End-to-end pipeline for Autonomous Explainable IDS.

Sequential pipeline:
DL -> Feature Gradients -> LLM -> Risk -> Agent

Phase 4 integration (optional):
- CombinedMemory retrieval with subset-first FG/embedding flow
- Retrieval metrics via MemoryEvaluator
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.loader import IDSDataLoader
from models.trainer import IDSModelTrainer
from explainability.feature_gradient_explainer import create_feature_gradient_explainer
from graph_correlation import (
    build_attack_graph,
    build_attack_profiles,
    get_top_correlated_pairs,
    validate_attack_graph,
)
from explainability.risk_scorer import create_risk_scorer
from llm.huggingface_client import create_huggingface_explainer
from agent.decision_agent import create_decision_agent
from memory import MemoryEvaluator
from memory.base_memory import BaseMemoryRetriever


RNG_SEED = 42


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
            raise ValueError("Embeddings/FG/labels must share same sample count.")

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


class IDSPipeline:

    def __init__(
        self,
        model_type: str = "hybrid",
        model_path: str = "saved_models/ids_model.keras",
        use_llm: bool = True,
        llm_model: str = "google/flan-t5-base",
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.use_llm = use_llm
        self.llm_model = llm_model

        self.data_loader = None
        self.trainer = None
        self.fg_explainer = None
        self.risk_scorer = None
        self.llm_explainer = None
        self.decision_agent = None
        self.attack_graph = None
        self.memory_retriever = None

        self.data = None
        self.label_mapping = None

        print("\n" + "=" * 70)
        print("AUTONOMOUS EXPLAINABLE INTRUSION DETECTION SYSTEM")
        print("=" * 70)

    # =====================================================
    # STEP 1: DATA
    # =====================================================

    def load_data(self):
        print("\n[STEP 1/6] Loading and Preprocessing Data")
        print("-" * 70)

        self.data_loader = IDSDataLoader(balanced_total_samples=None)
        self.data = self.data_loader.load_and_preprocess()
        self.label_mapping = self.data["label_mapping"]

        print("\n✓ Data loaded successfully")
        print(f"  Training samples: {self.data['X_train'].shape[0]}")
        print(f"  Test samples: {self.data['X_test'].shape[0]}")
        print(f"  Features: {self.data['num_features']}")
        print(f"  Classes: {self.data['num_classes']}")

    # =====================================================
    # STEP 2: MODEL
    # =====================================================

    def train_or_load_model(self, force_retrain: bool = False, epochs: int = 5, batch_size: Optional[int] = 128):
        print("\n[STEP 2/6] Deep Learning Model")
        print("-" * 70)
        print(f"Detected device: {IDSModelTrainer.detect_device()}")

        if batch_size is None:
            batch_size = IDSModelTrainer.recommended_batch_size()

        model_exists = os.path.exists(self.model_path)

        if model_exists and not force_retrain:
            print(f"Loading existing model from: {self.model_path}")
            self.trainer = IDSModelTrainer(model_type=self.model_type, model_save_path=self.model_path)
            self.trainer.model = IDSModelTrainer.load_model(self.model_path)
            self.trainer.evaluate(self.data["X_test"], self.data["y_test"])
        else:
            print(f"Training new {self.model_type.upper()} model...")
            self.trainer = IDSModelTrainer(model_type=self.model_type, model_save_path=self.model_path)
            self.trainer.train(
                self.data["X_train"],
                self.data["y_train"],
                self.data["X_test"],
                self.data["y_test"],
                epochs=epochs,
                batch_size=batch_size,
            )
            self.trainer.plot_training_history("training_history.png")
            self.trainer.get_detailed_report(self.data["X_test"], self.data["y_test"], self.label_mapping)

        print("\n✓ Model ready for inference")

    # =====================================================
    # STEP 3: FEATURE GRADIENTS
    # =====================================================

    def initialize_explainability(self):
        print("\n[STEP 3/6] Initializing Explainability (Feature Gradients)")
        print("-" * 70)

        self.fg_explainer = create_feature_gradient_explainer(self.trainer.model, self.data["feature_names"])
        print("\n✓ Feature Gradient explainer initialized")

    def build_graph_correlation_layer(self, max_samples: int = 500, threshold: float = 0.7):
        print("\n[STEP 3.1/6] Building Graph Correlation Layer")
        print("-" * 70)

        x_eval = self.data["X_test"][:max_samples]
        y_eval = self.data["y_test"][:max_samples]

        class_profiles = build_attack_profiles(
            X_eval=x_eval,
            y_eval=y_eval,
            fg_importance=self.fg_explainer.feature_importance,
        )

        self.attack_graph = build_attack_graph(class_profiles=class_profiles, threshold=threshold)
        validate_attack_graph(self.attack_graph)

        nodes = self.attack_graph.number_of_nodes()
        edges = self.attack_graph.number_of_edges()
        print(f"Graph nodes: {nodes}")
        print(f"Graph edges: {edges}")

        top_pairs = get_top_correlated_pairs(self.attack_graph, top_k=5)
        if top_pairs:
            print("Top correlated class pairs:")
            for a, b, w in top_pairs:
                a_name = self.label_mapping.get(a, str(a))
                b_name = self.label_mapping.get(b, str(b))
                print(f"  - {a_name} <-> {b_name}: {w:.4f}")
        else:
            print("No correlated pairs above threshold.")

    # =====================================================
    # STEP 4: LLM
    # =====================================================

    def initialize_llm(self):
        print("\n[STEP 4/6] Initializing LLM Reasoning")
        print("-" * 70)

        if self.use_llm:
            try:
                self.llm_explainer = create_huggingface_explainer(model_name=self.llm_model, temperature=0.3)
                print(f"\n✓ LLM initialized: {self.llm_model}")
            except Exception as e:
                print(f"Could not initialize LLM: {e}")
                self.use_llm = False
        else:
            print("LLM disabled")

    # =====================================================
    # STEP 5: RISK
    # =====================================================

    def initialize_risk_scorer(self):
        print("\n[STEP 5/6] Initializing Risk Scorer")
        print("-" * 70)
        self.risk_scorer = create_risk_scorer(self.label_mapping)
        print("\n✓ Risk scorer initialized")

    # =====================================================
    # STEP 6: AGENT
    # =====================================================

    def initialize_agent(self):
        print("\n[STEP 6/6] Initializing Decision Agent")
        print("-" * 70)
        self.decision_agent = create_decision_agent()
        print("\n✓ Decision agent initialized")

    # =====================================================
    # PHASE 4: COMBINED MEMORY (OPTIONAL)
    # =====================================================

    @staticmethod
    def _sample_subset(X: np.ndarray, y: np.ndarray, max_samples: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        n = int(X.shape[0])
        k = min(n, int(max_samples))
        if n > k:
            idx = rng.choice(n, size=k, replace=False)
        else:
            idx = np.arange(n)
        return np.asarray(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int32)

    def _compute_fg_matrix(self, explainer, X: np.ndarray) -> np.ndarray:
        fg_vectors: List[np.ndarray] = []
        total = X.shape[0]
        for i in range(total):
            fg_vectors.append(explainer.feature_importance(X[i]))
            if (i + 1) % 500 == 0:
                print(f"Computed FG vectors: {i + 1}/{total}")
        return np.asarray(fg_vectors, dtype=np.float32)

    def run_combined_memory_evaluation(self, top_k: int = 5, max_fg_samples: int = 10000, batch_size: Optional[int] = None):
        print("\n[PHASE 4] CombinedMemory Retrieval Evaluation")
        print("-" * 70)

        if batch_size is None:
            batch_size = IDSModelTrainer.recommended_batch_size()

        rng = np.random.default_rng(RNG_SEED)
        X_train_sub, y_train_sub = self._sample_subset(self.data["X_train"], self.data["y_train"], max_fg_samples, rng)
        X_test_sub, y_test_sub = self._sample_subset(self.data["X_test"], self.data["y_test"], max_fg_samples, rng)

        print(f"Train total: {self.data['X_train'].shape[0]} | subset: {X_train_sub.shape[0]}")
        print(f"Test total:  {self.data['X_test'].shape[0]} | subset: {X_test_sub.shape[0]}")

        # Subset sampling happens before embedding extraction.
        train_embeddings = self.trainer.extract_embeddings(X_train_sub, batch_size=batch_size)
        test_embeddings = self.trainer.extract_embeddings(X_test_sub, batch_size=batch_size)
        train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
        test_embeddings = np.asarray(test_embeddings, dtype=np.float32)

        explainer = create_feature_gradient_explainer(self.trainer.model, self.data["feature_names"])
        train_fg = self._compute_fg_matrix(explainer, X_train_sub)
        test_fg = self._compute_fg_matrix(explainer, X_test_sub)

        class_profiles = build_attack_profiles(X_eval=X_train_sub, y_eval=y_train_sub, fg_importance=explainer.feature_importance)
        phase4_graph = build_attack_graph(class_profiles=class_profiles, threshold=0.7)
        validate_attack_graph(phase4_graph)

        memory = CombinedMemory(top_k=top_k, embedding_weight=0.5)
        memory.fit(embeddings=train_embeddings, fg_vectors=train_fg, labels=y_train_sub)
        self.memory_retriever = memory

        y_pred = self.trainer.predict(X_test_sub, return_probabilities=False).astype(np.int32)

        evaluator = MemoryEvaluator(top_k=top_k)
        contexts: List[Dict[str, object]] = []
        retrieval_times: List[float] = []

        for i in range(X_test_sub.shape[0]):
            t0 = time.perf_counter()
            ctx = memory.retrieve(
                query_embedding=test_embeddings[i],
                query_fg=test_fg[i],
                predicted_class=int(y_pred[i]),
                top_k=top_k,
            )
            retrieval_times.append(time.perf_counter() - t0)
            contexts.append(ctx)

        metrics = evaluator.evaluate_contexts(contexts, y_test_sub)
        metrics["retrieval_time_sec"] = float(np.mean(retrieval_times)) if retrieval_times else 0.0

        print("\n" + "=" * 64)
        print("COMBINED MEMORY EVALUATION SUMMARY")
        print("=" * 64)
        print(f"Top-1 Retrieval Accuracy : {metrics['top1_retrieval_accuracy']:.4f}")
        print(f"Top-5 Retrieval Accuracy : {metrics['top5_retrieval_accuracy']:.4f}")
        print(f"Class Purity             : {metrics['class_purity']:.4f}")
        print(f"Retrieval Entropy        : {metrics['retrieval_entropy']:.4f}")
        print(f"Avg Cosine Similarity    : {metrics['avg_cosine_similarity']:.4f}")
        print(f"Retrieval Time (sec)     : {metrics['retrieval_time_sec']:.6f}")
        print("=" * 64)

        return metrics

    # =====================================================
    # RUN
    # =====================================================

    def run_pipeline(
        self,
        num_samples: int = 5,
        force_retrain: bool = False,
        enable_memory: bool = False,
        top_k: int = 5,
        max_fg_samples: int = 10000,
        batch_size: Optional[int] = None,
    ):
        self.load_data()
        self.train_or_load_model(force_retrain=force_retrain, batch_size=batch_size)
        self.initialize_explainability()
        self.build_graph_correlation_layer()

        if enable_memory:
            self.run_combined_memory_evaluation(top_k=top_k, max_fg_samples=max_fg_samples, batch_size=batch_size)

        self.initialize_llm()
        self.initialize_risk_scorer()
        self.initialize_agent()

        print("\n" + "=" * 70)
        print("PIPELINE READY - PROCESSING SAMPLES")
        print("=" * 70)

        results = []
        for i in range(min(num_samples, len(self.data["X_test"]))):
            results.append(self.process_sample(i, top_k=top_k if enable_memory else 5))

        self._save_results(results)
        return results

    # =====================================================
    # SAMPLE PROCESSING
    # =====================================================

    def process_sample(self, sample_index: int, top_k: int = 5):
        X_sample = self.data["X_test"][sample_index : sample_index + 1]
        y_true = self.data["y_test"][sample_index]
        true_label = self.label_mapping[y_true]

        predictions = self.trainer.predict(X_sample)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        attack_type = self.label_mapping[predicted_class]

        fg_explanation = self.fg_explainer.explain_prediction(X_sample, top_k=10)
        sample_fg = np.asarray(fg_explanation["importance_values_all"], dtype=np.float32)
        sample_embedding = self.trainer.extract_embeddings(X_sample)

        memory_context = {
            "top_k_labels": [],
            "similarity_scores": [],
            "class_distribution": {},
            "avg_similarity": 0.0,
        }
        if self.memory_retriever is not None:
            memory_context = self.memory_retriever.retrieve(
                query_embedding=sample_embedding[0],
                query_fg=sample_fg,
                predicted_class=predicted_class,
                top_k=top_k,
            )

        risk_result = self.risk_scorer.compute_risk_score(
            attack_type,
            confidence,
            fg_explanation["total_abs_importance"],
        )

        llm_explanation = None
        if self.use_llm and self.llm_explainer:
            llm_explanation = self.llm_explainer.explain_prediction(
                attack_type,
                confidence,
                risk_result["risk_score"],
                risk_result["severity_category"],
                fg_explanation["top_features"],
            )

        decision = self.decision_agent.decide_action(
            attack_type,
            confidence,
            risk_result["risk_score"],
            risk_result["severity_category"],
            llm_explanation,
        )

        return {
            "sample_index": sample_index,
            "true_label": true_label,
            "attack_type": attack_type,
            "confidence": confidence,
            "memory_context": memory_context,
            "risk_score": risk_result["risk_score"],
            "severity": risk_result["severity_category"],
            "agent_decision": decision["action"],
        }

    # =====================================================
    # SAVE
    # =====================================================

    def _save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ids_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {filename}")


# =====================================================
# MAIN
# =====================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--model", type=str, default="hybrid")
    parser.add_argument("--model_path", type=str, default="saved_models/ids_model.keras")
    parser.add_argument("--no-llm", action="store_true")

    # New Phase 4 integration flags.
    parser.add_argument("--enable_memory", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_fg_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    np.random.seed(RNG_SEED)

    pipeline = IDSPipeline(
        model_type=args.model,
        model_path=args.model_path,
        use_llm=not args.no_llm,
    )

    results = pipeline.run_pipeline(
        num_samples=args.samples,
        force_retrain=args.retrain,
        enable_memory=args.enable_memory,
        top_k=args.top_k,
        max_fg_samples=args.max_fg_samples,
        batch_size=args.batch_size,
    )

    print("\nPIPELINE COMPLETE")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()
