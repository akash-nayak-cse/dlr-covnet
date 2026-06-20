# data_loader.py
# Dataset mapping, class-weight calculation, and batch generator

import os
import numpy as np
import cv2
from sklearn.utils.class_weight import compute_class_weight


def create_data_mapping(data_path, label_file):
    """
    Parse a label text file into a {filename: label} dict.
    Expected format per line:  <filename> <positive|negative>

    Returns:
        dict[str, int]  — 1 for positive, 0 for negative
    """
    label_dict = {}
    with open(label_file, "r") as f:
        for line in f:
            filename, label = line.strip().split()
            label_dict[filename] = 1 if label.lower() == "positive" else 0
    return label_dict


def calculate_class_weights(label_dict):
    """
    Compute balanced inverse-frequency class weights.

    Returns:
        dict[int, float]  — {0: weight_neg, 1: weight_pos}
    """
    y = np.array(list(label_dict.values()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y,
    )
    return dict(enumerate(weights))


def custom_generator(data_path, label_dict, batch_size, class_weights=None, shuffle=True):
    """
    Infinite generator that yields (images, labels) or
    (images, labels, sample_weights) batches of normalised grayscale images.

    Args:
        data_path    (str)       : directory containing image files
        label_dict   (dict)      : {filename: int_label}
        batch_size   (int)       : number of samples per batch
        class_weights(dict|None) : if provided, yields per-sample weights
        shuffle      (bool)      : shuffle at the start of every epoch

    Yields:
        (np.ndarray, np.ndarray)           if class_weights is None
        (np.ndarray, np.ndarray, np.ndarray) otherwise
    """
    filenames = list(label_dict.keys())
    labels    = list(label_dict.values())

    while True:
        if shuffle:
            idx       = np.random.permutation(len(filenames))
            filenames = [filenames[i] for i in idx]
            labels    = [labels[i]    for i in idx]

        for i in range(0, len(filenames), batch_size):
            batch_files  = filenames[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            batch_images  = []
            batch_weights = []

            for filename, label in zip(batch_files, batch_labels):
                img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                batch_images.append(img)
                if class_weights is not None:
                    batch_weights.append(class_weights[label])

            if class_weights is not None:
                yield (
                    np.array(batch_images),
                    np.array(batch_labels),
                    np.array(batch_weights),
                )
            else:
                yield np.array(batch_images), np.array(batch_labels)
