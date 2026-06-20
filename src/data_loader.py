"""
Data loading utilities for DLR-CovNet.

This module provides helper functions for dataset mapping,
inverse class frequency weighting, and batch generation.

The complete data preprocessing pipeline accompanying the
published work is intentionally omitted from this repository.
"""

import os
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import config


def create_data_mapping(data_path, label_file):
    """
    Creates a filename-to-label mapping from dataset annotations.

    Returns:
        dict[str, int]
    """
    label_dict = {}

    with open(label_file, "r") as f:
        for line in f:
            filename, label = line.strip().split()
            label_dict[filename] = (
                1 if label.lower() == "positive" else 0
            )

    return label_dict


def calculate_class_weights(label_dict):
    """
    Computes inverse-frequency class weights.
    """

    labels = np.array(list(label_dict.values()))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )

    return dict(enumerate(weights))


def custom_generator(
    data_path,
    label_dict,
    batch_size,
    class_weights=None,
    shuffle=True,
):
    """
    Generates normalized grayscale image batches.
    """

    filenames = list(label_dict.keys())
    labels = list(label_dict.values())

    while True:

        if shuffle:
            indices = np.random.permutation(len(filenames))

            filenames = [filenames[i] for i in indices]
            labels = [labels[i] for i in indices]

        for start in range(0, len(filenames), batch_size):

            batch_files = filenames[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]

            images = []
            sample_weights = []

            for filename, label in zip(batch_files, batch_labels):

                image = cv2.imread(
                    os.path.join(data_path, filename),
                    cv2.IMREAD_GRAYSCALE,
                )

                # Standardize image size
                image = cv2.resize(
                    image,
                    config.INPUT_SHAPE[:2],
                )

                # Normalize pixel intensities
                image = image.astype(np.float32) / 255.0

                # Expand channel dimension
                image = np.expand_dims(image, axis=-1)

                images.append(image)

                if class_weights is not None:
                    sample_weights.append(
                        class_weights[label]
                    )

            images = np.array(images)
            batch_labels = np.array(batch_labels)

            if class_weights is not None:

                yield (
                    images,
                    batch_labels,
                    np.array(sample_weights),
                )

            else:

                yield (
                    images,
                    batch_labels,
                )