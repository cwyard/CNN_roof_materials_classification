"""Data loading, brightness filtering and dataset splitting utilities."""

from pathlib import Path

import cv2
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.preprocessing import apply_clahe_to_channel, create_clahe


def list_channel_files(red_pattern, green_pattern, blue_pattern):
    """Return sorted patch files for the red, green and blue channels."""
    files_red = sorted(Path(red_pattern).parent.glob(Path(red_pattern).name))
    files_green = sorted(Path(green_pattern).parent.glob(Path(green_pattern).name))
    files_blue = sorted(Path(blue_pattern).parent.glob(Path(blue_pattern).name))

    if not (len(files_red) == len(files_green) == len(files_blue)):
        raise ValueError("The three channel folders must contain the same number of image patches.")
    if len(files_red) == 0:
        raise FileNotFoundError("No image patches were found. Please check the input paths.")

    return files_red, files_green, files_blue


def parse_patch_metadata(file_path):
    """Extract the patch identifier and class label from a patch file name."""
    stem = Path(file_path).stem
    patch_id = stem.split("_")[0][3:]
    class_label = stem.split("_")[1]
    return patch_id, class_label


def read_channel(file_path, apply_clahe=False, clahe=None):
    """Read one image channel, replace invalid values and optionally apply CLAHE."""
    image = io.imread(file_path)
    image = np.nan_to_num(image, copy=True, nan=0.0, posinf=None, neginf=None)
    image = image.astype(np.uint8, copy=False)

    if apply_clahe:
        image = apply_clahe_to_channel(image, clahe)

    return image


def stack_rgb_patch(blue_path, green_path, red_path, apply_clahe=False, clahe=None):
    """Load and stack the blue, green and red channels into a three-band image tensor."""
    img_blue = read_channel(blue_path, apply_clahe=apply_clahe, clahe=clahe)
    img_green = read_channel(green_path, apply_clahe=apply_clahe, clahe=clahe)
    img_red = read_channel(red_path, apply_clahe=apply_clahe, clahe=clahe)
    return np.dstack([img_blue, img_green, img_red]).astype(np.float32)


def remap_class_label(class_label):
    """Map the original class labels to the 12-class scheme used for training."""
    if class_label == "12":
        return 11
    return int(class_label)


def load_dataset(
    files_blue,
    files_green,
    files_red,
    labels,
    max_reflectance,
    brightness_min,
    brightness_max,
    apply_clahe=False,
):
    """Load image patches, optionally apply CLAHE, filter brightness and normalise."""
    x_values = []
    y_values = []
    patch_ids = []
    clahe = create_clahe() if apply_clahe else None

    for blue_path, green_path, red_path in zip(files_blue, files_green, files_red):
        image_tensor = stack_rgb_patch(
            blue_path,
            green_path,
            red_path,
            apply_clahe=apply_clahe,
            clahe=clahe,
        )

        # Brightness is derived from the value channel after conversion to HSV.
        hsv_image = cv2.cvtColor(image_tensor[:, :, :3], cv2.COLOR_BGR2HSV)
        brightness = hsv_image[:, :, 2] / max_reflectance
        median_brightness = np.median(brightness)

        # Exclude patches that are too dark, shaded, too bright or overexposed.
        if brightness_min < median_brightness < brightness_max:
            patch_id, class_label = parse_patch_metadata(red_path)
            if class_label not in labels:
                raise ValueError(f"Unexpected class label '{class_label}' in {red_path}.")

            x_values.append(image_tensor / max_reflectance)
            y_values.append(remap_class_label(class_label))
            patch_ids.append(patch_id)

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.int64),
        np.asarray(patch_ids),
    )


def augment_with_mirror_symmetries(x_train, y_train):
    """Augment the training set using horizontal, vertical and combined flips."""
    x_augmented = x_train.tolist()
    y_augmented = y_train.tolist()

    for image, label in zip(x_train, y_train):
        x_augmented.extend([
            np.flip(image, axis=0),
            np.flip(image, axis=1),
            np.flip(np.flip(image, axis=0), axis=1),
        ])
        y_augmented.extend([label, label, label])

    return np.asarray(x_augmented, dtype=np.float32), np.asarray(y_augmented, dtype=np.int64)


def create_train_test_split(
    x_values,
    y_values,
    test_size,
    random_seed,
):
    """Split the dataset into training and test subsets without augmentation."""
    return train_test_split(
        x_values,
        y_values,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_values,
    )


def one_hot_encode_labels(y_train_raw, y_test_raw, n_classes):
    """One-hot encode raw integer labels after optional augmentation."""
    y_train = keras.utils.to_categorical(y_train_raw, n_classes)
    y_test = keras.utils.to_categorical(y_test_raw, n_classes)
    return y_train, y_test
