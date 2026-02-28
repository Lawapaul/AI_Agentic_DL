"""
Model training and evaluation module.

Handles:
- Training with early stopping and checkpointing
- Model evaluation
- Prediction generation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from models.model_factory import get_model
import matplotlib.pyplot as plt


class IDSModelTrainer:
    """Handles training and evaluation of IDS models dynamically."""

    def __init__(self, model_type="hybrid", model_save_path='saved_models/ids_model.keras'):
        """
        Initialize trainer with model type.

        Args:
            model_type: cnn / hybrid / resnet / lstm / etc.
            model_save_path: Path to save trained model
        """
        self.model_type = model_type
        self.model = None
        self.model_save_path = model_save_path
        self.history = None
        self.embedding_model = None

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    @staticmethod
    def detect_device():
        """Return active training device type for Colab-friendly logging."""
        gpus = tf.config.list_physical_devices("GPU")
        return "GPU" if gpus else "CPU"

    @staticmethod
    def recommended_batch_size():
        """Choose a conservative default batch size based on hardware."""
        return 256 if tf.config.list_physical_devices("GPU") else 128

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, X_train, y_train, X_test, y_test,
              epochs=50, batch_size=128):

        print(f"\n=== Training IDS {self.model_type.upper()} Model ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_test.shape[0]}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")

        # Ensure LSTM/CNN-compatible 3D input shape
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        print("Input shape after reshape:", X_train.shape)

        num_classes = len(np.unique(y_train))
        input_shape = (X_train.shape[1], 1)

        # Build model dynamically
        self.model = get_model(
            model_name=self.model_type,
            input_shape=input_shape,
            num_classes=num_classes
        )

        # 🔥 IMPORTANT FIX: Compile here
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n=== Training Complete ===")

        # Evaluate
        self.evaluate(X_test, y_test)

        return self.history

    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(self, X_test, y_test):

        print("\n=== Evaluating Model ===")

        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        return {
            'loss': test_loss,
            'accuracy': test_acc
        }

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X, return_probabilities=True):

        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = self.model.predict(X, verbose=0)

        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)

    # =====================================================
    # EMBEDDINGS (PENULTIMATE LAYER)
    # =====================================================

    def get_embedding_model(self):
        """
        Build a model that outputs penultimate activations.
        This is used by Phase 4 memory retrieval and does not alter prediction.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if self.embedding_model is None:
            # Prefer explicit embedding layer when available (Phase 4 contract).
            try:
                embedding_output = self.model.get_layer("embedding_layer").output
            except ValueError:
                if len(self.model.layers) < 2:
                    raise ValueError("Model must have at least two layers for penultimate embeddings.")
                embedding_output = self.model.layers[-2].output

            self.embedding_model = keras.Model(
                inputs=self.model.input,
                outputs=embedding_output,
                name=f"{self.model.name}_embedding_model",
            )
        return self.embedding_model

    def extract_embeddings(self, X, batch_size=256):
        """Extract penultimate-layer embeddings for samples X."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        emb_model = self.get_embedding_model()
        embeddings = emb_model.predict(X, batch_size=batch_size, verbose=0)
        return np.asarray(embeddings, dtype=np.float32)

    # =====================================================
    # DETAILED REPORT
    # =====================================================

    def get_detailed_report(self, X_test, y_test, label_mapping):

        y_pred = self.predict(X_test, return_probabilities=False)
        target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]

        print("\n=== Detailed Classification Report ===")
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)

        return report

    # =====================================================
    # PLOT HISTORY
    # =====================================================

    def plot_training_history(self, save_path='training_history.png'):

        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        plt.close()

    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save_model(self, path=None):
        save_path = path or self.model_save_path
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

    @staticmethod
    def load_model(path):
        print(f"Loading model from: {path}")
        return keras.models.load_model(path)


if __name__ == "__main__":
    print("Model trainer module loaded successfully")
