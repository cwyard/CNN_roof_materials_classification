"""Training utilities for individual CNNs and model ensembles."""

import numpy as np

from .evaluation import evaluate_predictions, plot_training_metric
from .model import build_cnn_model


def train_single_model(
    model_index,
    x_train,
    x_test,
    y_train,
    y_test,
    y_test_raw,
    input_shape,
    n_classes,
    batch_size,
    epochs,
    validation_split,
    figures_dir,
    loss="categorical_crossentropy",
    class_names=None,
):
    """Train and evaluate one CNN model."""
    print(f"\nTraining model {model_index}")
    model = build_cnn_model(input_shape=input_shape, n_classes=n_classes, loss=loss)
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    evaluate_predictions(
        y_test_raw,
        predicted_labels,
        figures_dir,
        file_stem=f"model_{model_index}",
        title_prefix=f"Model {model_index}",
        class_names=class_names,
    )
    plot_training_metric(history, "loss", figures_dir, model_index)

    return predictions, model, history


def train_ensemble(
    n_models,
    x_train,
    x_test,
    y_train,
    y_test,
    y_test_raw,
    input_shape,
    n_classes,
    batch_size,
    epochs,
    validation_split,
    figures_dir,
    loss="categorical_crossentropy",
    class_names=None,
):
    """Train an ensemble of CNN models and collect predictions and histories."""
    predictions_list = []
    models = []
    histories = []

    for model_index in range(1, n_models + 1):
        predictions, model, history = train_single_model(
            model_index=model_index,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            y_test_raw=y_test_raw,
            input_shape=input_shape,
            n_classes=n_classes,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            figures_dir=figures_dir,
            loss=loss,
            class_names=class_names,
        )
        predictions_list.append(predictions)
        models.append(model)
        histories.append(history)

    return predictions_list, models, histories


def average_ensemble_probabilities(predictions_list):
    """Average class probabilities across ensemble members."""
    ensemble_predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
    return ensemble_predictions / ensemble_predictions.sum(axis=1, keepdims=True)


def save_models(models, models_dir):
    """Save trained Keras models to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_index, model in enumerate(models, start=1):
        model.save(models_dir / f"model_{model_index}.h5")
    print(f"Saved {len(models)} models to {models_dir}")
