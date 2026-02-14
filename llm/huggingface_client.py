"""
HuggingFace LLM client for IDS explanations.
Replaces Ollama for cloud/Colab deployment.
"""

from transformers import pipeline
import torch


class HuggingFaceExplainer:
    """LLM explainer using HuggingFace models."""
    
    def __init__(self, model_name="google/flan-t5-base", temperature=0.3):
        """
        Initialize HuggingFace LLM.
        
        Args:
            model_name: HuggingFace model name
            temperature: Sampling temperature
        """
        print(f"\nLoading HuggingFace model: {model_name}...")
        
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
            max_length=512
        )
        self.temperature = temperature
        device_name = 'GPU' if device == 0 else 'CPU'
        print(f"✓ Model loaded on {device_name}")
    
    def explain_prediction(self, attack_type, confidence, risk_score, severity, top_features):
        """
        Generate explanation for a prediction.
        
        Args:
            attack_type: Predicted attack type
            confidence: Model confidence
            risk_score: Computed risk score
            severity: Severity category
            top_features: List of top SHAP features
            
        Returns:
            dict: Explanation results
        """
        # Create concise prompt
        feature_str = ", ".join([f"{f['feature_name']}" for f in top_features[:3]])
        
        prompt = f"""Analyze this network intrusion detection:
Attack Type: {attack_type}
Confidence: {confidence:.1%}
Risk Score: {risk_score:.2f}
Severity: {severity}
Key Features: {feature_str}

Provide a brief security analysis and recommended action:"""
        
        # Generate explanation
        result = self.generator(
            prompt,
            max_length=200,
            do_sample=True,
            temperature=self.temperature
        )
        
        explanation = result[0]['generated_text']
        
        return {
            'raw_explanation': explanation,
            'attack_type': attack_type,
            'confidence': confidence,
            'risk_assessment': f"{severity} risk - {attack_type}"
        }
    
    def explain_risk(self, attack_type, risk_score, severity):
        """
        Generate risk-focused explanation.
        
        Args:
            attack_type: Attack type
            risk_score: Risk score
            severity: Severity level
            
        Returns:
            str: Risk explanation
        """
        prompt = f"""Explain the security risk:
Attack: {attack_type}
Risk Score: {risk_score:.2f}/10
Severity: {severity}

Brief risk analysis:"""
        
        result = self.generator(
            prompt,
            max_length=150,
            do_sample=True,
            temperature=self.temperature
        )
        
        return result[0]['generated_text']


def create_huggingface_explainer(model_name="google/flan-t5-base", temperature=0.3):
    """
    Create HuggingFace explainer instance.
    
    Args:
        model_name: HuggingFace model name
        temperature: Sampling temperature
        
    Returns:
        HuggingFaceExplainer instance
    """
    return HuggingFaceExplainer(model_name, temperature)


if __name__ == "__main__":
    # Test the explainer
    print("Testing HuggingFace Explainer...")
    explainer = create_huggingface_explainer()
    
    # Test explanation
    test_features = [
        {'feature_name': 'Flow Duration', 'shap_value': 0.5},
        {'feature_name': 'Total Fwd Packets', 'shap_value': 0.3},
        {'feature_name': 'Flow Bytes/s', 'shap_value': 0.2}
    ]
    
    result = explainer.explain_prediction(
        attack_type="SSH-Bruteforce",
        confidence=0.95,
        risk_score=8.5,
        severity="CRITICAL",
        top_features=test_features
    )
    
    print("\n=== Test Result ===")
    print(f"Explanation: {result['raw_explanation']}")
    print("\n✓ Test complete")
