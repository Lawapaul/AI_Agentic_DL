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
        self.feature_names = list(feature_names)  # Ensure list
        self.explainer = None

        print("\n=== Initializing SHAP Explainer ===")
        print(f"Background data shape: {background_data.shape}")
        print(f"Number of features: {len(feature_names)}")

        # Initialize GradientExplainer
        self.explainer = shap.GradientExplainer(
            self.model,
            self.background_data
        )

        print("SHAP GradientExplainer initialized successfully")

    # -------------------------------------------------------------
    # INTERNAL SAFE SHAPE HANDLER
    # -------------------------------------------------------------
    def _extract_class_shap(self, shap_values, predicted_class):
        """
        Safely extract SHAP values for predicted class
        and flatten to (features,) shape.
        """

        # Multi-class (list of arrays)
        if isinstance(shap_values, list):
            if len(shap_values) > predicted_class:
                class_shap = shap_values[predicted_class]
            else:
                class_shap = shap_values[0]
        else:
            class_shap = shap_values

        # Expected shape: (1, features, 1) or (1, features)
        class_shap = np.array(class_shap)

        # Remove batch dimension
        if class_shap.ndim >= 2:
            class_shap = class_shap[0]

        # Remove last singleton dimension
        class_shap = np.squeeze(class_shap)

        # Ensure 1D
        class_shap = class_shap.reshape(-1)

        return class_shap

    # -------------------------------------------------------------
    # SINGLE SAMPLE EXPLANATION
    # -------------------------------------------------------------
    def explain_prediction(self, sample, top_k=10):
        """
        Generate SHAP explanation for a single prediction.

        Args:
            sample: Shape (1, features, 1)
            top_k: Number of top features

        Returns:
            dict
        """

        # Ensure NumPy input
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()

        # Compute SHAP
        shap_values = self.explainer.shap_values(sample)

        # Predict class
        prediction = self.model.predict(sample, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])

        # Extract correct class SHAP safely
        class_shap_values = self._extract_class_shap(
            shap_values,
            predicted_class
        )

        abs_shap_values = np.abs(class_shap_values)

        # Safety: prevent overflow top_k
        top_k = min(top_k, len(abs_shap_values))

        # Get top indices
        top_indices = np.argsort(abs_shap_values)[-top_k:][::-1]

        top_features = []

        for idx in top_indices:
            idx = int(idx)  # Force scalar integer

            # Safety: bounds check
            if idx >= len(self.feature_names):
                continue

            top_features.append({
                'feature_name': self.feature_names[idx],
                'feature_index': idx,
                'shap_value': float(class_shap_values[idx]),
                'abs_shap_value': float(abs_shap_values[idx])
            })

        total_abs_shap = float(np.sum(abs_shap_values))

        explanation = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_features': top_features,
            'total_abs_shap': total_abs_shap,
            'shap_values_all': class_shap_values.tolist()
        }

        return explanation

    # -------------------------------------------------------------
    # BATCH EXPLANATION
    # -------------------------------------------------------------
    def explain_batch(self, samples, top_k=10):

        if isinstance(samples, tf.Tensor):
            samples = samples.numpy()

        print(f"\nGenerating SHAP explanations for {samples.shape[0]} samples...")

        explanations = []

        for i in range(samples.shape[0]):
            sample = samples[i:i+1]
            explanation = self.explain_prediction(sample, top_k)
            explanations.append(explanation)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{samples.shape[0]} samples")

        print("SHAP explanations complete")
        return explanations

    # -------------------------------------------------------------
    # FEATURE IMPORTANCE SUMMARY
    # -------------------------------------------------------------
    def get_feature_importance_summary(self, samples, max_samples=100):

        if isinstance(samples, tf.Tensor):
            samples = samples.numpy()

        n_samples = min(len(samples), max_samples)
        sample_subset = samples[:n_samples]

        print(f"\nComputing feature importance summary for {n_samples} samples...")

        shap_values = self.explainer.shap_values(sample_subset)

        all_abs_shap = []

        for class_shap in shap_values:
            class_shap = np.array(class_shap)

            # Remove last dim if exists
            class_shap = np.squeeze(class_shap)

            # Ensure shape (samples, features)
            if class_shap.ndim == 3:
                class_shap = class_shap[:, :, 0]

            all_abs_shap.append(np.abs(class_shap))

        stacked_shap = np.stack(all_abs_shap, axis=0)
        mean_abs_shap = np.mean(stacked_shap, axis=(0, 1))

        feature_ranking = np.argsort(mean_abs_shap)[::-1]

        importance_summary = {
            'feature_importance': [
                {
                    'rank': i + 1,
                    'feature_name': self.feature_names[int(idx)],
                    'feature_index': int(idx),
                    'mean_abs_shap': float(mean_abs_shap[int(idx)])
                }
                for i, idx in enumerate(feature_ranking)
            ]
        }

        print("Feature importance summary complete")
        return importance_summary


# Convenience function
def create_shap_explainer(model, background_data, feature_names):
    return SHAPExplainer(model, background_data, feature_names)


if __name__ == "__main__":
    print("SHAP explainer module loaded successfully")