"""CNN architecture definition."""

from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn_model(input_shape, n_classes, loss="categorical_crossentropy"):
    """Build the CNN architecture used for rooftop material classification."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(
                n_classes,
                activity_regularizer=regularizers.L2(1e-5),
                activation="softmax",
            ),
        ]
    )
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    return model
