"""Transfer-learning utilities based on MobileNetV2 pre-trained on ImageNet."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .evaluation import evaluate_predictions, plot_training_metric


def build_mobilenetv2_model(
    input_shape,
    n_classes,
    target_size=(224, 224),
    learning_rate=1e-4,
    dropout_rate=0.3,
    dense_units=128,
    loss="categorical_crossentropy",
):
    """Build a MobileNetV2 transfer-learning classifier.

    The input patches remain 64 x 64 pixels. A resizing layer is included inside
    the model so that the pre-trained MobileNetV2 backbone receives 224 x 224
    inputs. The backbone is loaded with ImageNet weights and frozen by default.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Resizing(target_size[0], target_size[1], name="resize_to_mobilenetv2")(inputs)
    x = layers.Lambda(lambda image: preprocess_input(image * 255.0), name="mobilenetv2_preprocess")(x)

    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(target_size[0], target_size[1], input_shape[-1]),
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_ImageNet_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def enable_mobilenetv2_fine_tuning(model, fine_tune_at=100, learning_rate=1e-5, loss="categorical_crossentropy"):
    """Unfreeze the upper part of the MobileNetV2 backbone for optional fine-tuning."""
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def train_mobilenetv2_baseline(
    x_train,
    x_test,
    y_train,
    y_test,
    y_test_raw,
    input_shape,
    n_classes,
    target_size,
    batch_size,
    epochs,
    validation_split,
    figures_dir,
    learning_rate=1e-4,
    dropout_rate=0.3,
    dense_units=128,
    fine_tune=False,
    fine_tune_at=100,
    fine_tune_epochs=20,
    fine_tune_learning_rate=1e-5,
    loss="categorical_crossentropy",
    class_names=None,
):
    """Train and evaluate the MobileNetV2 ImageNet transfer-learning baseline."""
    print("\nTraining MobileNetV2 ImageNet transfer-learning baseline")
    model = build_mobilenetv2_model(
        input_shape=input_shape,
        n_classes=n_classes,
        target_size=target_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        loss=loss,
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1,
    )

    if fine_tune:
        print("\nFine-tuning the upper MobileNetV2 layers")
        model = enable_mobilenetv2_fine_tuning(
            model,
            fine_tune_at=fine_tune_at,
            learning_rate=fine_tune_learning_rate,
            loss=loss,
        )
        fine_tune_history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=fine_tune_epochs,
            validation_split=validation_split,
            verbose=1,
        )
    else:
        fine_tune_history = None

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"MobileNetV2 test loss: {test_loss:.4f}")
    print(f"MobileNetV2 test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    evaluation_outputs = evaluate_predictions(
        y_test_raw,
        predicted_labels,
        figures_dir,
        file_stem="mobilenetv2_imagenet",
        title_prefix="MobileNetV2 ImageNet",
        class_names=class_names,
    )
    plot_training_metric(history, "loss", figures_dir, "mobilenetv2_imagenet")

    return probabilities, predicted_labels, model, history, fine_tune_history, evaluation_outputs
