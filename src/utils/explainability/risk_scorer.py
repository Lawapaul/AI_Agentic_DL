"""
Risk scoring module for IDS predictions.

Computes reproducible risk scores based on:
- DL model confidence
- SHAP-based anomaly score
- Rule-based severity weights
"""

import numpy as np


class RiskScorer:
    """Computes risk scores for intrusion detection predictions."""
    
    # Severity weights for different attack types
    # These are domain-specific and should be tuned based on security policies
    SEVERITY_WEIGHTS = {
        'BENIGN': 0.0,
        'NORMAL': 0.0,
        'DOS': 0.9,
        'DDOS': 0.95,
        'PROBE': 0.6,
        'PROBING': 0.6,
        'R2L': 0.8,
        'U2R': 0.85,
        'BOTNET': 0.9,
        'PORTSCAN': 0.5,
        'BRUTEFORCE': 0.75,
        'INFILTRATION': 0.9,
        'WEBATTACK': 0.7,
        'DEFAULT': 0.5  # Default for unknown attack types
    }
    
    def __init__(self, label_mapping, 
                 confidence_weight=0.4, 
                 shap_weight=0.3, 
                 severity_weight=0.3):
        """
        Initialize risk scorer.
        
        Args:
            label_mapping: Dictionary mapping class indices to attack names
            confidence_weight: Weight for DL confidence in risk score
            shap_weight: Weight for SHAP anomaly score in risk score
            severity_weight: Weight for severity in risk score
        """
        self.label_mapping = label_mapping
        self.confidence_weight = confidence_weight
        self.shap_weight = shap_weight
        self.severity_weight = severity_weight
        
        # Validate weights sum to 1
        total_weight = confidence_weight + shap_weight + severity_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        print("\n=== Risk Scorer Initialized ===")
        print(f"Confidence weight: {confidence_weight}")
        print(f"SHAP weight: {shap_weight}")
        print(f"Severity weight: {severity_weight}")
    
    def get_severity_weight(self, attack_type):
        """
        Get severity weight for an attack type.
        
        Args:
            attack_type: Name of the attack type
            
        Returns:
            float: Severity weight [0, 1]
        """
        # Normalize attack type to uppercase
        attack_type_upper = attack_type.upper()
        
        # Check for exact match
        if attack_type_upper in self.SEVERITY_WEIGHTS:
            return self.SEVERITY_WEIGHTS[attack_type_upper]
        
        # Check for partial match (e.g., "DoS Hulk" contains "DOS")
        for key in self.SEVERITY_WEIGHTS:
            if key in attack_type_upper or attack_type_upper in key:
                return self.SEVERITY_WEIGHTS[key]
        
        # Return default if no match
        return self.SEVERITY_WEIGHTS['DEFAULT']
    
    def normalize_shap_score(self, total_abs_shap, max_shap=100.0):
        """
        Normalize SHAP anomaly score to [0, 1].
        
        Args:
            total_abs_shap: Sum of absolute SHAP values
            max_shap: Maximum expected SHAP value for normalization
            
        Returns:
            float: Normalized SHAP score [0, 1]
        """
        # Use sigmoid-like normalization to handle varying ranges
        normalized = total_abs_shap / (total_abs_shap + max_shap)
        return min(normalized, 1.0)
    
    def compute_risk_score(self, attack_type, confidence, total_abs_shap):
        """
        Compute risk score for a prediction.
        
        Formula:
        risk_score = (confidence * w1) + (normalized_shap * w2) + (severity * w3)
        
        Args:
            attack_type: Predicted attack type name
            confidence: DL model confidence (softmax probability)
            total_abs_shap: Sum of absolute SHAP values
            
        Returns:
            dict: Risk score and components
        """
        # Get severity weight
        severity = self.get_severity_weight(attack_type)
        
        # Normalize SHAP score
        shap_score = self.normalize_shap_score(total_abs_shap)
        
        # Compute weighted risk score
        risk_score = (
            confidence * self.confidence_weight +
            shap_score * self.shap_weight +
            severity * self.severity_weight
        )
        
        # Determine severity category
        severity_category = self._get_severity_category(risk_score)
        
        return {
            'risk_score': float(risk_score),
            'severity_category': severity_category,
            'components': {
                'confidence': float(confidence),
                'shap_score': float(shap_score),
                'severity_weight': float(severity),
                'confidence_contribution': float(confidence * self.confidence_weight),
                'shap_contribution': float(shap_score * self.shap_weight),
                'severity_contribution': float(severity * self.severity_weight)
            }
        }
    
    def _get_severity_category(self, risk_score):
        """
        Map risk score to severity category.
        
        Args:
            risk_score: Computed risk score [0, 1]
            
        Returns:
            str: Severity category
        """
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        elif risk_score < 0.85:
            return "High"
        else:
            return "Critical"
    
    def batch_compute_risk_scores(self, predictions, shap_explanations):
        """
        Compute risk scores for multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries
            shap_explanations: List of SHAP explanation dictionaries
            
        Returns:
            list: List of risk score dictionaries
        """
        risk_scores = []
        
        for pred, shap_exp in zip(predictions, shap_explanations):
            attack_type = self.label_mapping[pred['predicted_class']]
            confidence = pred['confidence']
            total_abs_shap = shap_exp['total_abs_shap']
            
            risk_score = self.compute_risk_score(
                attack_type, confidence, total_abs_shap
            )
            risk_scores.append(risk_score)
        
        return risk_scores


def create_risk_scorer(label_mapping):
    """
    Convenience function to create risk scorer.
    
    Args:
        label_mapping: Dictionary mapping class indices to attack names
        
    Returns:
        RiskScorer instance
    """
    return RiskScorer(label_mapping)


if __name__ == "__main__":
    # Test risk scorer
    print("Testing Risk Scorer...")
    
    test_mapping = {
        0: 'BENIGN',
        1: 'DoS',
        2: 'Probe',
        3: 'R2L',
        4: 'U2R'
    }
    
    scorer = create_risk_scorer(test_mapping)
    
    # Test cases
    test_cases = [
        ('BENIGN', 0.95, 5.0),
        ('DoS', 0.92, 50.0),
        ('Probe', 0.75, 30.0),
        ('R2L', 0.88, 45.0)
    ]
    
    print("\n=== Test Cases ===")
    for attack_type, confidence, shap_score in test_cases:
        result = scorer.compute_risk_score(attack_type, confidence, shap_score)
        print(f"\nAttack: {attack_type}, Confidence: {confidence}, SHAP: {shap_score}")
        print(f"Risk Score: {result['risk_score']:.4f}")
        print(f"Severity: {result['severity_category']}")
