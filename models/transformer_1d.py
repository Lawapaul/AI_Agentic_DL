from tensorflow import keras
from tensorflow.keras import layers


def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):

    # Multi-head attention
    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(x, x)

    attention = layers.Dropout(dropout)(attention)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention)

    # Feed-forward network
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)

    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    return x


def build_transformer_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)

    x = layers.Dense(64)(inputs)  # project features

    x = transformer_block(x, head_size=32, num_heads=2, ff_dim=128)
    x = transformer_block(x, head_size=32, num_heads=2, ff_dim=128)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="IDS_TRANSFORMER_1D")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model