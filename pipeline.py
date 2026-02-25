"""
End-to-end pipeline for Autonomous Explainable IDS.

Now supports multiple architectures:
- cnn
- hybrid
- (future: resnet, lstm, gru, transformer)

Sequential pipeline:
DL → Feature Gradients → LLM → Risk → Agent
"""

import os
import numpy as np
import json
from datetime import datetime

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
from memory import (
    EmbeddingKNNMemory,
    FeatureGradientKNNMemory,
    GraphAwareMemory,
    PrototypeMemory,
)


class IDSPipeline:

    def __init__(
        self,
        model_type="hybrid",
        model_path="saved_models/ids_model.keras",
        use_llm=True,
        llm_model="google/flan-t5-base",
        memory_strategy="embedding_knn",
        memory_top_k=5,
        use_memory=True,
    ):
        """
        Args:
            model_type: cnn / hybrid / resnet / etc.
        """

        self.model_type = model_type
        self.model_path = model_path
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.use_memory = use_memory
        self.memory_strategy = memory_strategy
        self.memory_top_k = int(memory_top_k)

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

        # Phase 4 requirement: full dataset loading by default (no balanced sampling).
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

    def train_or_load_model(self, force_retrain=False, epochs=5, batch_size=128):

        print("\n[STEP 2/6] Deep Learning Model")
        print("-" * 70)
        print(f"Detected device: {IDSModelTrainer.detect_device()}")

        if batch_size is None:
            batch_size = IDSModelTrainer.recommended_batch_size()

        model_exists = os.path.exists(self.model_path)

        if model_exists and not force_retrain:
            print(f"Loading existing model from: {self.model_path}")

            self.trainer = IDSModelTrainer(
                model_type=self.model_type,
                model_save_path=self.model_path,
            )

            self.trainer.model = IDSModelTrainer.load_model(self.model_path)

            self.trainer.evaluate(
                self.data["X_test"], self.data["y_test"]
            )

        else:
            print(f"Training new {self.model_type.upper()} model...")

            self.trainer = IDSModelTrainer(
                model_type=self.model_type,
                model_save_path=self.model_path,
            )

            self.trainer.train(
                self.data["X_train"],
                self.data["y_train"],
                self.data["X_test"],
                self.data["y_test"],
                epochs=epochs,
                batch_size=batch_size,
            )

            self.trainer.plot_training_history("training_history.png")

            self.trainer.get_detailed_report(
                self.data["X_test"],
                self.data["y_test"],
                self.label_mapping,
            )

        print("\n✓ Model ready for inference")

    # =====================================================
    # STEP 3: FEATURE GRADIENTS
    # =====================================================

    def initialize_explainability(self, background_samples=100):

        print("\n[STEP 3/6] Initializing Explainability (Feature Gradients)")
        print("-" * 70)

        self.fg_explainer = create_feature_gradient_explainer(
            self.trainer.model,
            self.data["feature_names"],
        )

        print("\n✓ Feature Gradient explainer initialized")

    def build_graph_correlation_layer(self, max_samples=500, threshold=0.7):

        print("\n[STEP 3.1/6] Building Graph Correlation Layer")
        print("-" * 70)

        x_eval = self.data["X_test"][:max_samples]
        y_eval = self.data["y_test"][:max_samples]

        class_profiles = build_attack_profiles(
            X_eval=x_eval,
            y_eval=y_eval,
            fg_importance=self.fg_explainer.feature_importance,
        )

        self.attack_graph = build_attack_graph(
            class_profiles=class_profiles,
            threshold=threshold,
        )

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

    def _compute_fg_matrix(self, X):
        """Compute per-sample FG vectors for memory retrieval."""
        fg_vectors = []
        total = X.shape[0]
        for i in range(total):
            fg_vectors.append(self.fg_explainer.feature_importance(X[i]))
            if (i + 1) % 500 == 0:
                print(f"Computed FG vectors: {i + 1}/{total}")
        return np.asarray(fg_vectors, dtype=np.float32)

    def initialize_memory_retrieval(self, embedding_batch_size=None):
        """
        Phase 4 retrieval-only memory layer.
        This does not alter prediction outputs or risk scoring.
        """
        print("\n[STEP 3.2/6] Initializing Memory Retrieval")
        print("-" * 70)

        if not self.use_memory:
            self.memory_retriever = None
            print("Memory retrieval disabled")
            return

        if embedding_batch_size is None:
            embedding_batch_size = IDSModelTrainer.recommended_batch_size()

        train_embeddings = self.trainer.extract_embeddings(
            self.data["X_train"],
            batch_size=embedding_batch_size,
        )
        train_labels = self.data["y_train"]

        strategy = self.memory_strategy.lower().strip()
        if strategy == "embedding_knn":
            self.memory_retriever = EmbeddingKNNMemory(top_k=self.memory_top_k)
            self.memory_retriever.fit(embeddings=train_embeddings, labels=train_labels)
        elif strategy == "fg_knn":
            train_fg = self._compute_fg_matrix(self.data["X_train"])
            self.memory_retriever = FeatureGradientKNNMemory(top_k=self.memory_top_k)
            self.memory_retriever.fit(fg_vectors=train_fg, labels=train_labels)
        elif strategy == "prototype":
            self.memory_retriever = PrototypeMemory(top_k=self.memory_top_k)
            self.memory_retriever.fit(embeddings=train_embeddings, labels=train_labels)
        elif strategy == "graph_aware":
            self.memory_retriever = GraphAwareMemory(top_k=self.memory_top_k)
            self.memory_retriever.fit(
                embeddings=train_embeddings,
                labels=train_labels,
                attack_graph=self.attack_graph,
            )
        else:
            raise ValueError(
                "Unknown memory strategy. Use one of: "
                "embedding_knn, fg_knn, prototype, graph_aware"
            )

        print(f"\n✓ Memory retrieval initialized: {strategy}")

    # =====================================================
    # STEP 4: LLM
    # =====================================================

    def initialize_llm(self):

        print("\n[STEP 4/6] Initializing LLM Reasoning")
        print("-" * 70)

        if self.use_llm:
            try:
                self.llm_explainer = create_huggingface_explainer(
                    model_name=self.llm_model,
                    temperature=0.3,
                )
                print(f"\n✓ LLM initialized: {self.llm_model}")
            except Exception as e:
                print(f"⚠ Could not initialize LLM: {e}")
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
    # RUN
    # =====================================================

    def run_pipeline(self, num_samples=5, force_retrain=False):

        self.load_data()
        self.train_or_load_model(force_retrain=force_retrain)
        self.initialize_explainability()
        self.build_graph_correlation_layer()
        self.initialize_memory_retrieval()
        self.initialize_llm()
        self.initialize_risk_scorer()
        self.initialize_agent()

        print("\n" + "=" * 70)
        print("PIPELINE READY - PROCESSING SAMPLES")
        print("=" * 70)

        results = []

        for i in range(min(num_samples, len(self.data["X_test"]))):
            result = self.process_sample(i)
            results.append(result)

        self._save_results(results)

        return results

    # =====================================================
    # SAMPLE PROCESSING
    # =====================================================

    def process_sample(self, sample_index):

        X_sample = self.data["X_test"][sample_index : sample_index + 1]
        y_true = self.data["y_test"][sample_index]
        true_label = self.label_mapping[y_true]

        predictions = self.trainer.predict(X_sample)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        attack_type = self.label_mapping[predicted_class]

        fg_explanation = self.fg_explainer.explain_prediction(
            X_sample, top_k=10
        )
        sample_fg = np.asarray(
            fg_explanation["importance_values_all"],
            dtype=np.float32,
        )
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
                top_k=self.memory_top_k,
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
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--memory-strategy",
        type=str,
        default="embedding_knn",
        choices=["embedding_knn", "fg_knn", "prototype", "graph_aware"],
    )
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--no-memory", action="store_true")

    args = parser.parse_args()

    pipeline = IDSPipeline(
        model_type=args.model,
        use_llm=not args.no_llm,
        memory_strategy=args.memory_strategy,
        memory_top_k=args.memory_top_k,
        use_memory=not args.no_memory,
    )

    results = pipeline.run_pipeline(
        num_samples=args.samples,
        force_retrain=args.retrain,
    )

    print("\nPIPELINE COMPLETE")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()
