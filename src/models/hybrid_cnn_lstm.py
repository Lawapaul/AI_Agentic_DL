import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):

    inputs = layers.Input(shape=input_shape)

    # CNN Block
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # LSTM Block (explicit embedding exposure for Phase 4 memory retrieval)
    x = layers.LSTM(128, name="embedding_layer")(x)


    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def extract_embedding(model, X, batch_size=256):
    """
    Extract embeddings from the named embedding layer.
    Kept lightweight for Colab and Phase 4 experiments.
    """
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    embedding_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding_layer").output,
    )
    return embedding_model.predict(X, batch_size=batch_size, verbose=0)
