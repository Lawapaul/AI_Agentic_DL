"""
End-to-end pipeline for Autonomous Explainable IDS.

This module orchestrates:
1. Data loading and preprocessing
2. Deep Learning prediction (1D CNN)
3. SHAP explainability
4. LLM reasoning (HuggingFace)
5. Risk scoring
6. Decision agent action execution

Sequential pipeline: DL → SHAP → LLM → Risk → Agent
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

# Import all components
from data.loader import IDSDataLoader
from models.cnn_model import create_ids_model
from models.trainer import IDSModelTrainer
from explainability.shap_explainer import create_shap_explainer
from explainability.risk_scorer import create_risk_scorer
from llm.huggingface_client import create_huggingface_explainer
from agent.decision_agent import create_decision_agent


class IDSPipeline:
    """
    End-to-end Autonomous Explainable IDS Pipeline.
    
    Maintains strict separation of concerns:
    - DL predicts
    - SHAP explains features
    - LLM interprets
    - Agent automates actions
    """
    
    def __init__(self, model_path='saved_models/ids_cnn.keras', 
                 use_llm=True, llm_model='google/flan-t5-base'):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to save/load trained model
            use_llm: Whether to use HuggingFace LLM for reasoning
            llm_model: HuggingFace model name
        """
        self.model_path = model_path
        self.use_llm = use_llm
        self.llm_model = llm_model
        
        # Components (initialized during run)
        self.data_loader = None
        self.model = None
        self.trainer = None
        self.shap_explainer = None
        self.risk_scorer = None
        self.llm_explainer = None
        self.decision_agent = None
        
        # Data and metadata
        self.data = None
        self.label_mapping = None
        
        print("\n" + "="*70)
        print("AUTONOMOUS EXPLAINABLE INTRUSION DETECTION SYSTEM")
        print("="*70)
    
    def load_data(self):
        """Load and preprocess the IDS dataset."""
        print("\n[STEP 1/6] Loading and Preprocessing Data")
        print("-" * 70)
        
        self.data_loader = IDSDataLoader()
        self.data = self.data_loader.load_and_preprocess()
        self.label_mapping = self.data['label_mapping']
        
        print(f"\n✓ Data loaded successfully")
        print(f"  Training samples: {self.data['X_train'].shape[0]}")
        print(f"  Test samples: {self.data['X_test'].shape[0]}")
        print(f"  Features: {self.data['num_features']}")
        print(f"  Classes: {self.data['num_classes']}")
    
    def train_or_load_model(self, force_retrain=False, epochs=1, batch_size=128):
        """
        Train a new model or load existing one.
        
        Args:
            force_retrain: If True, train new model even if one exists
            epochs: Training epochs (default: 1 for quick training)
            batch_size: Batch size for training
        """
        print("\n[STEP 2/6] Deep Learning Model")
        print("-" * 70)
        
        # Check if model exists
        model_exists = os.path.exists(self.model_path)
        
        if model_exists and not force_retrain:
            print(f"Loading existing model from: {self.model_path}")
            self.model = IDSModelTrainer.load_model(self.model_path)
            self.trainer = IDSModelTrainer(self.model, self.model_path)
            
            # Evaluate on test set
            self.trainer.evaluate(self.data['X_test'], self.data['y_test'])
        
        else:
            print("Training new model...")
            
            # Create model
            input_shape = (self.data['num_features'], 1)
            self.model = create_ids_model(input_shape, self.data['num_classes'])
            
            # Train model
            self.trainer = IDSModelTrainer(self.model, self.model_path)
            self.trainer.train(
                self.data['X_train'], self.data['y_train'],
                self.data['X_test'], self.data['y_test'],
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Plot training history
            self.trainer.plot_training_history('training_history.png')
            
            # Get detailed report
            self.trainer.get_detailed_report(
                self.data['X_test'], 
                self.data['y_test'], 
                self.label_mapping
            )
        
        print(f"\n✓ Model ready for inference")
    
    def initialize_explainability(self, background_samples=100):
        """
        Initialize SHAP explainer.
        
        Args:
            background_samples: Number of samples for SHAP background
        """
        print("\n[STEP 3/6] Initializing Explainability (SHAP)")
        print("-" * 70)
        
        # Use subset of training data as background
        background_data = self.data['X_train'][:background_samples]
        
        self.shap_explainer = create_shap_explainer(
            self.model,
            background_data,
            self.data['feature_names']
        )
        
        print(f"\n✓ SHAP explainer initialized")
    
    def initialize_llm(self):
        """Initialize LLM explainer (HuggingFace)."""
        print("\n[STEP 4/6] Initializing LLM Reasoning (HuggingFace)")
        print("-" * 70)
        
        if self.use_llm:
            try:
                self.llm_explainer = create_huggingface_explainer(
                    model_name=self.llm_model,
                    temperature=0.3
                )
                print(f"\n✓ LLM explainer initialized with {self.llm_model}")
            except Exception as e:
                print(f"\n⚠ Warning: Could not initialize HuggingFace LLM: {e}")
                print("Continuing without LLM reasoning...")
                self.use_llm = False
        else:
            print("LLM reasoning disabled")
    
    def initialize_risk_scorer(self):
        """Initialize risk scorer."""
        print("\n[STEP 5/6] Initializing Risk Scorer")
        print("-" * 70)
        
        self.risk_scorer = create_risk_scorer(self.label_mapping)
        
        print(f"\n✓ Risk scorer initialized")
    
    def initialize_agent(self):
        """Initialize decision agent."""
        print("\n[STEP 6/6] Initializing Decision Agent")
        print("-" * 70)
        
        self.decision_agent = create_decision_agent(use_llm_reasoning=False)
        
        print(f"\n✓ Decision agent initialized")
    
    def process_sample(self, sample_index, verbose=True):
        """
        Process a single sample through the complete pipeline.
        
        Args:
            sample_index: Index of test sample to process
            verbose: If True, print detailed output
            
        Returns:
            dict: Complete structured output
        """
        # Get sample
        X_sample = self.data['X_test'][sample_index:sample_index+1]
        y_true = self.data['y_test'][sample_index]
        true_label = self.label_mapping[y_true]
        
        if verbose:
            print("\n" + "="*70)
            print(f"PROCESSING SAMPLE {sample_index}")
            print("="*70)
            print(f"True Label: {true_label}")
        
        # 1. Deep Learning Prediction
        if verbose:
            print("\n[1] Deep Learning Prediction...")
        
        predictions = self.trainer.predict(X_sample, return_probabilities=True)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        attack_type = self.label_mapping[predicted_class]
        
        if verbose:
            print(f"  Predicted: {attack_type}")
            print(f"  Confidence: {confidence:.4f}")
        
        # 2. SHAP Explanation
        if verbose:
            print("\n[2] SHAP Explainability...")
        
        shap_explanation = self.shap_explainer.explain_prediction(X_sample, top_k=10)
        top_features = shap_explanation['top_features']
        total_abs_shap = shap_explanation['total_abs_shap']
        
        if verbose:
            print(f"  Top features: {[f['feature_name'] for f in top_features[:3]]}")
            print(f"  Total SHAP: {total_abs_shap:.4f}")
        
        # 3. Risk Scoring
        if verbose:
            print("\n[3] Risk Scoring...")
        
        risk_result = self.risk_scorer.compute_risk_score(
            attack_type, confidence, total_abs_shap
        )
        risk_score = risk_result['risk_score']
        severity = risk_result['severity_category']
        
        if verbose:
            print(f"  Risk Score: {risk_score:.4f}")
            print(f"  Severity: {severity}")
        
        # 4. LLM Reasoning
        llm_explanation = None
        if self.use_llm and self.llm_explainer:
            if verbose:
                print("\n[4] LLM Reasoning...")
            
            llm_explanation = self.llm_explainer.explain_prediction(
                attack_type, confidence, risk_score, severity, top_features
            )
            
            if verbose:
                print(f"  Explanation generated")
        elif verbose:
            print("\n[4] LLM Reasoning... (skipped)")
        
        # 5. Decision Agent
        if verbose:
            print("\n[5] Decision Agent...")
        
        decision = self.decision_agent.decide_action(
            attack_type, confidence, risk_score, severity, llm_explanation
        )
        
        if verbose:
            print(f"  Action: {decision['action']}")
            print(f"  Type: {decision['action_type']}")
        
        # Construct final output
        output = {
            'sample_index': sample_index,
            'true_label': true_label,
            'attack_type': attack_type,
            'confidence': confidence,
            'top_features': [
                {
                    'name': f['feature_name'],
                    'shap_value': f['shap_value']
                }
                for f in top_features[:5]
            ],
            'llm_explanation': llm_explanation['raw_explanation'] if llm_explanation else 'N/A',
            'risk_score': risk_score,
            'severity': severity,
            'agent_decision': decision['action'],
            'action_taken': decision['execution_log']
        }
        
        return output
    
    def run_pipeline(self, num_samples=5, force_retrain=False):
        """
        Run the complete pipeline on test samples.
        
        Args:
            num_samples: Number of test samples to process
            force_retrain: Whether to force model retraining
            
        Returns:
            list: List of output dictionaries
        """
        # Initialize all components
        self.load_data()
        self.train_or_load_model(force_retrain=force_retrain)
        self.initialize_explainability()
        self.initialize_llm()
        self.initialize_risk_scorer()
        self.initialize_agent()
        
        print("\n" + "="*70)
        print("PIPELINE READY - PROCESSING SAMPLES")
        print("="*70)
        
        # Process samples
        results = []
        for i in range(min(num_samples, len(self.data['X_test']))):
            result = self.process_sample(i, verbose=True)
            results.append(result)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ids_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Explainable IDS Pipeline')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to process')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--no-llm', action='store_true', help='Disable HuggingFace LLM')
    parser.add_argument('--llm-model', type=str, default='google/flan-t5-base', help='HuggingFace model name')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = IDSPipeline(
        use_llm=not args.no_llm,
        llm_model=args.llm_model
    )
    
    results = pipeline.run_pipeline(
        num_samples=args.samples,
        force_retrain=args.retrain
    )
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"Processed {len(results)} samples successfully")


if __name__ == "__main__":
    main()
