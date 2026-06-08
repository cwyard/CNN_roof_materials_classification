"""Evaluation and plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_training_metric(history, metric, figures_dir, model_index):
    """Plot and save a training/validation metric curve."""
    train_metric = history.history[metric]
    validation_metric = history.history[f"val_{metric}"]
    epochs_range = range(1, len(train_metric) + 1)

    plt.figure()
    plt.plot(epochs_range, train_metric, label=f"train_{metric}")
    plt.plot(epochs_range, validation_metric, label=f"val_{metric}")
    plt.title(f"Training and validation {metric} - model {model_index}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"model_{model_index}_{metric}.png", dpi=300)
    plt.show()


def print_classification_summary(reference_labels, predicted_labels, class_names=None):
    """Print standard classification metrics and the per-class F1 scores."""
    print("Overall accuracy:", metrics.accuracy_score(reference_labels, predicted_labels))
    print(
        "Weighted precision:",
        metrics.precision_score(reference_labels, predicted_labels, average="weighted", zero_division=0),
    )
    print(
        "Weighted recall:",
        metrics.recall_score(reference_labels, predicted_labels, average="weighted", zero_division=0),
    )
    print(
        "Weighted F1 score:",
        metrics.f1_score(reference_labels, predicted_labels, average="weighted", zero_division=0),
    )
    print("\nPer-class classification report:")
    print(
        metrics.classification_report(
            reference_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0,
        )
    )


def save_classification_report(reference_labels, predicted_labels, figures_dir, file_stem, class_names=None):
    """Save precision, recall and F1 scores by class as a CSV file."""
    report = metrics.classification_report(
        reference_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    import pandas as pd

    report_table = pd.DataFrame(report).transpose()
    output_path = figures_dir / f"{file_stem}_classification_report.csv"
    report_table.to_csv(output_path, index=True)
    print(f"Saved classification report to {output_path}")
    return report_table


def plot_confusion_matrix(
    reference_labels,
    predicted_labels,
    figures_dir,
    file_stem,
    title,
    class_names=None,
    normalise=None,
):
    """Plot and save a confusion matrix.

    Set ``normalise`` to None for raw counts, "true" for recall-normalised
    rows, or "pred" for precision-normalised columns.
    """
    confusion_matrix = metrics.confusion_matrix(
        reference_labels,
        predicted_labels,
        normalize=normalise,
    )

    if normalise == "pred":
        print("Class-wise precision:", np.diag(confusion_matrix))
    elif normalise == "true":
        print("Class-wise recall:", np.diag(confusion_matrix))

    plt.figure(figsize=(8, 7))
    plt.imshow(confusion_matrix)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("Reference class")
    plt.colorbar()

    tick_positions = np.arange(len(class_names)) if class_names is not None else np.arange(len(np.unique(reference_labels)))
    if class_names is not None:
        plt.xticks(tick_positions, class_names, rotation=45, ha="right")
        plt.yticks(tick_positions, class_names)

    plt.tight_layout()
    suffix = "raw" if normalise is None else f"normalised_{normalise}"
    plt.savefig(figures_dir / f"{file_stem}_confusion_matrix_{suffix}.png", dpi=300)
    plt.show()

    return confusion_matrix


def evaluate_predictions(
    reference_labels,
    predicted_labels,
    figures_dir,
    file_stem,
    title_prefix,
    class_names=None,
):
    """Run the standard evaluation workflow for a set of predictions."""
    print_classification_summary(reference_labels, predicted_labels, class_names=class_names)
    report_table = save_classification_report(
        reference_labels,
        predicted_labels,
        figures_dir,
        file_stem=file_stem,
        class_names=class_names,
    )
    raw_confusion_matrix = plot_confusion_matrix(
        reference_labels,
        predicted_labels,
        figures_dir,
        file_stem=file_stem,
        title=f"{title_prefix} - raw confusion matrix",
        class_names=class_names,
        normalise=None,
    )
    precision_confusion_matrix = plot_confusion_matrix(
        reference_labels,
        predicted_labels,
        figures_dir,
        file_stem=file_stem,
        title=f"{title_prefix} - precision-normalised confusion matrix",
        class_names=class_names,
        normalise="pred",
    )
    recall_confusion_matrix = plot_confusion_matrix(
        reference_labels,
        predicted_labels,
        figures_dir,
        file_stem=file_stem,
        title=f"{title_prefix} - recall-normalised confusion matrix",
        class_names=class_names,
        normalise="true",
    )
    return {
        "classification_report": report_table,
        "raw_confusion_matrix": raw_confusion_matrix,
        "precision_confusion_matrix": precision_confusion_matrix,
        "recall_confusion_matrix": recall_confusion_matrix,
    }


def compute_brier_scores(reference_labels, probabilities, n_classes):
    """Compute one-vs-rest Brier scores for all classes."""
    brier_scores = {}
    for class_id in range(n_classes):
        reference_binary = (reference_labels == class_id).astype(int)
        class_probability = probabilities[:, class_id]
        score = metrics.brier_score_loss(reference_binary, class_probability)
        brier_scores[class_id] = score
        print(f"Brier score for class {class_id}: {score:.4f}")
    return brier_scores


def plot_roc_curves(reference_labels, probabilities, n_classes, figures_dir):
    """Plot one-vs-rest ROC curves for all classes."""
    for class_id in range(n_classes):
        reference_binary = (reference_labels == class_id).astype(int)
        class_probability = probabilities[:, class_id]
        fpr, tpr, _ = metrics.roc_curve(reference_binary, class_probability)
        auc = metrics.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.title(f"ROC curve - class {class_id}")
        plt.axis([0, 1, 0, 1])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"ROC_class_{class_id}.png", dpi=300)
        plt.show()
