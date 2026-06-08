"""Loss functions and class-weight utilities."""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def compute_balanced_class_weights(y_train_raw, n_classes):
    """Compute balanced class weights from the training labels only.

    The returned array has one weight per class and can be passed to the
    weighted categorical cross-entropy loss. Classes absent from the training
    split receive a weight of 0.0 to avoid indexing errors, although stratified
    splitting should normally prevent this situation.
    """
    class_ids = np.arange(n_classes)
    present_classes = np.unique(y_train_raw)
    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train_raw,
    )

    weights = np.zeros(n_classes, dtype=np.float32)
    weights[present_classes] = present_weights.astype(np.float32)

    print("Class weights used for weighted categorical cross-entropy:")
    for class_id, weight in enumerate(weights):
        print(f"  class {class_id}: {weight:.4f}")

    return weights


def weighted_categorical_crossentropy(class_weights):
    """Create a weighted categorical cross-entropy loss function.

    Parameters
    ----------
    class_weights : array-like
        One weight per class, ordered according to the one-hot encoded label
        columns.
    """
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        unweighted_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        sample_weights = tf.reduce_sum(y_true * weights, axis=-1)
        return unweighted_loss * sample_weights

    loss.__name__ = "weighted_categorical_crossentropy"
    return loss
