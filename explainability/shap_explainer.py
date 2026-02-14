"""
SHAP explainability module for CNN predictions.

This module:
- Uses SHAP to explain CNN predictions
- Extracts top contributing features
- Computes feature importance values
- SHAP runs AFTER the DL model prediction
"""

import numpy as np
import shap
import tensorflow as tf


class SHAPExplainer:
    """Handles SHAP-based explainability for IDS CNN model."""
    
    def __init__(self, model, background_data, feature_names):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained Keras model
            background_data: Background dataset for SHAP (subset of training data)
            feature_names: List of feature names
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        
        print("\n=== Initializing SHAP Explainer ===")
        print(f"Background data shape: {background_data.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Initialize SHAP GradientExplainer
        # GradientExplainer works well with deep learning models
        self.explainer = shap.GradientExplainer(
            model,
            background_data
        )
        
        print("SHAP GradientExplainer initialized successfully")
    
    def explain_prediction(self, sample, top_k=10):
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            sample: Single sample to explain (shape: (1, features, 1))
            top_k: Number of top features to return
            
        Returns:
            dict: Explanation with top features and their SHAP values
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(sample)
        
        # shap_values is a list (one array per class)
        # Get the predicted class
        prediction = self.model.predict(sample, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list) and len(shap_values) > predicted_class:
            # Multi-class: list of arrays, one per class
            class_shap_values = shap_values[predicted_class][0].squeeze()
        elif isinstance(shap_values, list) and len(shap_values) > 0:
            # Fallback: use first available class
            class_shap_values = shap_values[0][0].squeeze()
        else:
            # Single array format
            class_shap_values = shap_values[0].squeeze()
        
        # Get absolute values for ranking importance
        abs_shap_values = np.abs(class_shap_values)
        
        # Get top-k feature indices
        top_indices = np.argsort(abs_shap_values)[-top_k:][::-1]
        
        # Create explanation dictionary
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature_name': self.feature_names[idx],
                'feature_index': int(idx),
                'shap_value': float(class_shap_values[idx]),
                'abs_shap_value': float(abs_shap_values[idx])
            })
        
        # Compute total absolute SHAP value (for anomaly scoring)
        total_abs_shap = float(np.sum(abs_shap_values))
        
        explanation = {
            'predicted_class': int(predicted_class),
            'confidence': float(prediction[0][predicted_class]),
            'top_features': top_features,
            'total_abs_shap': total_abs_shap,
            'shap_values_all': class_shap_values.tolist()
        }
        
        return explanation
    
    def explain_batch(self, samples, top_k=10):
        """
        Generate SHAP explanations for multiple samples.
        
        Args:
            samples: Batch of samples (shape: (batch_size, features, 1))
            top_k: Number of top features to return per sample
            
        Returns:
            list: List of explanation dictionaries
        """
        print(f"\nGenerating SHAP explanations for {samples.shape[0]} samples...")
        
        explanations = []
        for i in range(samples.shape[0]):
            sample = samples[i:i+1]  # Keep batch dimension
            explanation = self.explain_prediction(sample, top_k)
            explanations.append(explanation)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{samples.shape[0]} samples")
        
        print("SHAP explanations complete")
        return explanations
    
    def get_feature_importance_summary(self, samples, max_samples=100):
        """
        Get overall feature importance across multiple samples.
        
        Args:
            samples: Samples to analyze
            max_samples: Maximum number of samples to use
            
        Returns:
            dict: Feature importance summary
        """
        # Limit samples for computational efficiency
        n_samples = min(len(samples), max_samples)
        sample_subset = samples[:n_samples]
        
        print(f"\nComputing feature importance summary for {n_samples} samples...")
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(sample_subset)
        
        # Average absolute SHAP values across all classes and samples
        # shap_values is a list of arrays, one per class
        all_abs_shap = []
        for class_shap in shap_values:
            # Shape: (samples, features, 1) -> squeeze last dim
            class_shap_squeezed = class_shap.squeeze(axis=-1)
            all_abs_shap.append(np.abs(class_shap_squeezed))
        
        # Stack and average
        stacked_shap = np.stack(all_abs_shap, axis=0)  # (classes, samples, features)
        mean_abs_shap = np.mean(stacked_shap, axis=(0, 1))  # Average over classes and samples
        
        # Rank features
        feature_ranking = np.argsort(mean_abs_shap)[::-1]
        
        importance_summary = {
            'feature_importance': [
                {
                    'rank': i + 1,
                    'feature_name': self.feature_names[idx],
                    'feature_index': int(idx),
                    'mean_abs_shap': float(mean_abs_shap[idx])
                }
                for i, idx in enumerate(feature_ranking)
            ]
        }
        
        print("Feature importance summary complete")
        return importance_summary


def create_shap_explainer(model, background_data, feature_names):
    """
    Convenience function to create SHAP explainer.
    
    Args:
        model: Trained Keras model
        background_data: Background dataset for SHAP
        feature_names: List of feature names
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, background_data, feature_names)


if __name__ == "__main__":
    print("SHAP explainer module loaded successfully")
