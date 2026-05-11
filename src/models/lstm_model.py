from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)

    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="IDS_LSTM")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model