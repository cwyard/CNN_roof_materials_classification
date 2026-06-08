"""Optional pre-processing utilities for image enhancement and orientation normalisation."""

import cv2
import numpy as np
from skimage.feature import hog


def create_clahe(clip_limit=2.0, tile_grid_size=(8, 8)):
    """Create a CLAHE transformer for local contrast enhancement."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)


def apply_clahe_to_channel(image, clahe=None):
    """Apply CLAHE to a single image channel."""
    if clahe is None:
        clahe = create_clahe()
    return clahe.apply(image)


def rotate_anisotropic_patches(
    x_values,
    n_channels,
    hog_orientations,
    hog_min_threshold,
    hog_max_threshold,
    hog_accumulation_threshold,
):
    """Rotate patches whose HOG profile suggests anisotropic texture."""
    x_rotated = x_values.copy()

    for index in range(x_rotated.shape[0]):
        # Convert the RGB patch into a panchromatic image using standard luminance weights.
        pan_image = (
            0.1140 * x_rotated[index, :, :, 0]
            + 0.5870 * x_rotated[index, :, :, 1]
            + 0.2989 * x_rotated[index, :, :, 2]
        )

        hog_features, _ = hog(
            pan_image,
            orientations=hog_orientations,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=True,
        )

        orientation_scores = np.zeros(hog_orientations)
        for i, value in enumerate(hog_features):
            orientation_scores[i % hog_orientations] += value

        # Count orientations whose HOG score lies within the threshold interval.
        accumulated_orientations = np.sum(
            (orientation_scores > hog_min_threshold)
            & (orientation_scores < hog_max_threshold)
        )

        if accumulated_orientations < hog_accumulation_threshold:
            height, width = pan_image.shape[:2]
            centre = (width / 2, height / 2)
            angle = np.argmax(orientation_scores) * (180 / hog_orientations)
            rotation_matrix = cv2.getRotationMatrix2D(centre, angle, scale=1)

            for channel in range(n_channels):
                x_rotated[index, :, :, channel] = cv2.warpAffine(
                    x_rotated[index, :, :, channel],
                    rotation_matrix,
                    (width, height),
                )

    return x_rotated
