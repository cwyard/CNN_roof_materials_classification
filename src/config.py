"""Configuration values used by the CNN training workflow.

Edit the paths and experiment-level parameters here before running the notebook.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------
# Replace this path with the folder containing the red, green and blue subfolders.
DATA_DIR = Path(r"X:/Projets/CASMATTELEv2/Stage_Hugo/data/Namur/p64/mod")

RED_PATTERN = DATA_DIR / "red" / "*.tif"
GREEN_PATTERN = DATA_DIR / "green" / "*.tif"
BLUE_PATTERN = DATA_DIR / "blue" / "*.tif"

# Replace this path with the directory where models and figures should be saved.
OUTPUT_BASE_DIR = Path(r"X:/Projets/CASMATTELEv2/Stage_Hugo/py/classification/DL")
EXPERIMENT_NAME = "DL_3_github"
OUTPUT_DIR = OUTPUT_BASE_DIR / EXPERIMENT_NAME
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 2

# -----------------------------------------------------------------------------
# Image and class parameters
# -----------------------------------------------------------------------------
N_CHANNELS = 3
N_BITS = 8
MAX_REFLECTANCE = 2 ** N_BITS
INPUT_SHAPE = (64, 64, N_CHANNELS)

# The original labelling scheme contains class 12, which is remapped to class 11
# because class 11 is absent from the dataset.
LABELS = [str(i) for i in range(13)]
N_CLASSES = 12

# -----------------------------------------------------------------------------
# Optional pre-processing
# -----------------------------------------------------------------------------
# If True, Contrast Limited Adaptive Histogram Equalisation is applied to each
# input channel during data loading.
USE_CLAHE = True

# If True, a HOG-based rotation is applied to normalise patches with anisotropic
# texture orientation before the training/test split.
USE_HOG = True

# -----------------------------------------------------------------------------
# Filtering and HOG parameters
# -----------------------------------------------------------------------------
BRIGHTNESS_MIN = 0.2
BRIGHTNESS_MAX = 0.9
HOG_ORIENTATIONS = 32
HOG_MIN_THRESHOLD = 5
HOG_MAX_THRESHOLD = 15
HOG_ACCUMULATION_THRESHOLD = 30

# -----------------------------------------------------------------------------
# Optional data augmentation
# -----------------------------------------------------------------------------
# If True, mirror symmetries are applied to the training set only, after the
# training/test split. The test set is never augmented.
USE_DATA_AUGMENTATION = True


# ----------------------------------------------------------------------------
# Optional class-imbalance handling
# ----------------------------------------------------------------------------
# If True, class weights are computed from the training labels and used in a
# weighted categorical cross-entropy loss. The test set is never used to compute
# these weights.
USE_WEIGHTED_CATEGORICAL_CROSSENTROPY = True

# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 100
N_MODELS = 12
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1


# -----------------------------------------------------------------------------
# Optional transfer-learning baseline: MobileNetV2 pre-trained on ImageNet
# -----------------------------------------------------------------------------
# If True, the notebook trains an additional MobileNetV2 transfer-learning
# baseline. This experiment is independent from the custom CNN ensemble.
RUN_MOBILENETV2_TEST = True

# MobileNetV2 accepts smaller inputs when include_top=False, but 224 x 224 is
# used here to follow the standard ImageNet pre-training configuration.
MOBILENETV2_INPUT_SIZE = (224, 224)
MOBILENETV2_BATCH_SIZE = 32
MOBILENETV2_EPOCHS = 50
MOBILENETV2_LEARNING_RATE = 1e-4
MOBILENETV2_DROPOUT_RATE = 0.3
MOBILENETV2_DENSE_UNITS = 128

# Fine-tuning is disabled by default to provide a conservative comparison based
# on frozen ImageNet features. It can be enabled after the classifier head has
# been trained.
MOBILENETV2_FINE_TUNE = False
MOBILENETV2_FINE_TUNE_AT = 100
MOBILENETV2_FINE_TUNE_EPOCHS = 20
MOBILENETV2_FINE_TUNE_LEARNING_RATE = 1e-5
MOBILENETV2_MODELS_DIR = MODELS_DIR / "mobilenetv2"

