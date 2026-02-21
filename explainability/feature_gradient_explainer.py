"""
Feature Gradient explainability module for IDS model predictions.

Uses TensorFlow GradientTape saliency:
- gradient(predicted_class_probability, input_features)
- absolute gradient as feature importance
"""

import numpy as np
import tensorflow as tf


class FeatureGradientExplainer:
    """Computes feature attributions using input gradients."""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)

        print("\n=== Initializing Feature Gradient Explainer ===")
        print(f"Number of features: {len(self.feature_names)}")

    def _as_3d(self, sample):
        """Ensure sample shape is (1, features, 1)."""
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()

        sample = np.asarray(sample, dtype=np.float32)

        if sample.ndim == 1:
            sample = sample.reshape(1, -1, 1)
        elif sample.ndim == 2:
            # (features, 1) or (1, features)
            if sample.shape[0] == 1:
                sample = sample.reshape(1, sample.shape[1], 1)
            else:
                sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        elif sample.ndim != 3:
            raise ValueError(f"Unsupported sample shape: {sample.shape}")

        return sample

    def feature_importance(self, sample):
        """
        Return per-feature absolute gradients for predicted class.

        Args:
            sample: input sample compatible with model input

        Returns:
            np.ndarray: shape (features,)
        """
        x = self._as_3d(sample)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            probs = self.model(x_tf, training=False)
            pred_class = tf.argmax(probs[0], axis=-1, output_type=tf.int32)
            target_prob = tf.gather(probs[0], pred_class)

        grads = tape.gradient(target_prob, x_tf)

        # Edge case safety: gradient can be None for disconnected graph paths.
        if grads is None:
            return np.zeros((x.shape[1],), dtype=np.float32)

        saliency = tf.abs(grads)[0, :, 0]
        return saliency.numpy().astype(np.float32)

    def explain_prediction(self, sample, top_k=10):
        """
        Build explanation payload for one sample.
        """
        x = self._as_3d(sample)

        probs = self.model.predict(x, verbose=0)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        importance = self.feature_importance(x)
        top_k = min(top_k, len(importance))
        top_indices = np.argsort(importance)[-top_k:][::-1]

        top_features = []
        for idx in top_indices:
            idx = int(idx)
            top_features.append(
                {
                    "feature_name": self.feature_names[idx],
                    "feature_index": idx,
                    "importance_value": float(importance[idx]),
                }
            )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_features": top_features,
            "total_abs_importance": float(np.sum(importance)),
            "importance_values_all": importance.tolist(),
        }

    def explain_batch(self, samples, top_k=10):
        if isinstance(samples, tf.Tensor):
            samples = samples.numpy()

        print(
            f"\nGenerating Feature Gradient explanations for {samples.shape[0]} samples..."
        )

        explanations = []
        for i in range(samples.shape[0]):
            sample = samples[i : i + 1]
            explanations.append(self.explain_prediction(sample, top_k=top_k))
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{samples.shape[0]} samples")

        print("Feature Gradient explanations complete")
        return explanations


def create_feature_gradient_explainer(model, feature_names):
    return FeatureGradientExplainer(model, feature_names)

