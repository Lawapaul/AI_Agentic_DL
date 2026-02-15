"""
End-to-end pipeline for Autonomous Explainable IDS.

Now supports multiple architectures:
- cnn
- hybrid
- (future: resnet, lstm, gru, transformer)

Sequential pipeline:
DL → SHAP → LLM → Risk → Agent
"""

import os
import numpy as np
import json
from datetime import datetime

from data.loader import IDSDataLoader
from models.trainer import IDSModelTrainer
from explainability.shap_explainer import create_shap_explainer
from explainability.risk_scorer import create_risk_scorer
from llm.huggingface_client import create_huggingface_explainer
from agent.decision_agent import create_decision_agent


class IDSPipeline:

    def __init__(
        self,
        model_type="cnn",
        model_path="saved_models/ids_model.keras",
        use_llm=True,
        llm_model="google/flan-t5-base",
    ):
        """
        Args:
            model_type: cnn / hybrid / resnet / etc.
        """

        self.model_type = model_type
        self.model_path = model_path
        self.use_llm = use_llm
        self.llm_model = llm_model

        self.data_loader = None
        self.trainer = None
        self.shap_explainer = None
        self.risk_scorer = None
        self.llm_explainer = None
        self.decision_agent = None

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

        self.data_loader = IDSDataLoader()
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
    # STEP 3: SHAP
    # =====================================================

    def initialize_explainability(self, background_samples=100):

        print("\n[STEP 3/6] Initializing Explainability (SHAP)")
        print("-" * 70)

        background_data = self.data["X_train"][:background_samples]

        self.shap_explainer = create_shap_explainer(
            self.trainer.model,
            background_data,
            self.data["feature_names"],
        )

        print("\n✓ SHAP explainer initialized")

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

        shap_explanation = self.shap_explainer.explain_prediction(
            X_sample, top_k=10
        )

        risk_result = self.risk_scorer.compute_risk_score(
            attack_type,
            confidence,
            shap_explanation["total_abs_shap"],
        )

        llm_explanation = None
        if self.use_llm and self.llm_explainer:
            llm_explanation = self.llm_explainer.explain_prediction(
                attack_type,
                confidence,
                risk_result["risk_score"],
                risk_result["severity_category"],
                shap_explanation["top_features"],
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
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--no-llm", action="store_true")

    args = parser.parse_args()

    pipeline = IDSPipeline(
        model_type=args.model,
        use_llm=not args.no_llm,
    )

    results = pipeline.run_pipeline(
        num_samples=args.samples,
        force_retrain=args.retrain,
    )

    print("\nPIPELINE COMPLETE")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()