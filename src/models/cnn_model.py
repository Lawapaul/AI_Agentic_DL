"""
1D CNN model architecture for network intrusion detection.

This module implements a research-grade deep learning model with:
- Modular CNN blocks
- Residual connections
- Batch normalization
- Sufficient depth for academic evaluation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# Note: tf.keras is the recommended import path


@tf.keras.utils.register_keras_serializable()
class CNNBlock(layers.Layer):
    """
    Modular CNN block with Conv1D, BatchNorm, ReLU, and Dropout.
    
    This reusable block forms the building unit of the network.
    """
    
    def __init__(self, filters, kernel_size=3, dropout_rate=0.3, **kwargs):
        """
        Initialize CNN block.
        
        Args:
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dropout_rate: Dropout probability
        """
        super(CNNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=None
        )
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        """Forward pass through the block."""
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ResidualCNNBlock(layers.Layer):
    """
    CNN block with residual connection for improved gradient flow.
    
    Implements: output = CNN(input) + input (with projection if needed)
    """
    
    def __init__(self, filters, kernel_size=3, dropout_rate=0.3, **kwargs):
        """
        Initialize residual CNN block.
        
        Args:
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dropout_rate: Dropout probability
        """
        super(ResidualCNNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        self.cnn_block = CNNBlock(filters, kernel_size, dropout_rate)
        self.projection = None  # Will be created if needed
    
    def build(self, input_shape):
        """Build the layer - create projection if input/output dims differ."""
        input_filters = input_shape[-1]
        
        if input_filters != self.filters:
            # Need projection to match dimensions
            self.projection = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same'
            )
        
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """Forward pass with residual connection."""
        x = self.cnn_block(inputs, training=training)
        
        # Add residual connection
        if self.projection is not None:
            residual = self.projection(inputs)
        else:
            residual = inputs
        
        return x + residual
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


def build_ids_cnn_model(input_shape, num_classes):
    """
    Build a research-grade 1D CNN for intrusion detection.
    
    Architecture:
    - Input layer
    - 4 Residual CNN blocks with increasing filters (64 -> 128 -> 256 -> 256)
    - Global Average Pooling
    - Dense layer with BatchNorm and Dropout
    - Output layer with softmax
    
    Args:
        input_shape: Shape of input data (features, 1)
        num_classes: Number of attack classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial CNN block
    x = CNNBlock(filters=64, kernel_size=3, dropout_rate=0.3)(inputs)
    
    # Residual CNN blocks with increasing complexity
    x = ResidualCNNBlock(filters=128, kernel_size=3, dropout_rate=0.3)(x)
    x = ResidualCNNBlock(filters=256, kernel_size=3, dropout_rate=0.3)(x)
    x = ResidualCNNBlock(filters=256, kernel_size=3, dropout_rate=0.4)(x)
    
    # Additional depth for academic rigor
    x = ResidualCNNBlock(filters=512, kernel_size=3, dropout_rate=0.4)(x)
    
    # Global pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers for classification
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer with softmax for multi-class classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='IDS_CNN')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Simplified to avoid batch size issues
    )
    
    return model


def create_ids_model(input_shape, num_classes, learning_rate=0.001):
    """
    Create and compile the complete IDS CNN model.
    
    Args:
        input_shape: Shape of input data (features, 1)
        num_classes: Number of attack classes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled keras model
    """
    print("\n=== Building IDS CNN Model ===")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = build_ids_cnn_model(input_shape, num_classes)
    model = compile_model(model, learning_rate)
    
    print("\nModel architecture:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing IDS CNN Model...")
    
    # Example: 41 features (typical for network flow data)
    test_input_shape = (41, 1)
    test_num_classes = 5  # e.g., Normal, DoS, Probe, R2L, U2R
    
    model = create_ids_model(test_input_shape, test_num_classes)
    
    print("\n=== Model Test Complete ===")
