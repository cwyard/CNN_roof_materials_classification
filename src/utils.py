"""General utilities for reproducible experiments."""

import random

import numpy as np
import tensorflow as tf


def set_random_seed(seed):
    """Set random seeds for Python, NumPy and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_output_directories(*directories):
    """Create output directories if they do not already exist."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
