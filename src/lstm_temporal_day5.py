import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear"]
emap = {e: i for i, e in enumerate(EMOTIONS)}


def demo_train():
    # Ejemplo mínimo de secuencias (cada secuencia es una lista de índices)
    X = np.array([
        [emap["happy"], emap["happy"], emap["neutral"], emap["happy"]],
        [emap["sad"], emap["neutral"], emap["sad"], emap["sad"]],
    ])

    y = np.array([emap["happy"], emap["sad"]])

    X = to_categorical(X, num_classes=len(EMOTIONS))
    y = to_categorical(y, num_classes=len(EMOTIONS))

    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(len(EMOTIONS), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X, y, epochs=10, verbose=1)


if __name__ == "__main__":
    demo_train()
