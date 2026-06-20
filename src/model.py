"""
DLR-CovNet: Dual-Scale Lightweight Residual Network

TensorFlow implementation of the proposed DLR-CovNet architecture
presented in the Springer publication.

The architecture employs iterative Residual Dual-Conv Pooling (RDCP)
blocks with parallel 3×3 and 5×5 convolutional kernels to learn
multi-scale feature representations for COVID-19 classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def rdcp_block(inputs, filters, kernel_size):
    """
    Residual Dual-Conv Pooling (RDCP) block.
    """

    shortcut = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        activation="relu",
    )(x)

    x = layers.Add()([x, shortcut])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    return x


def create_model(input_shape=(224, 224, 1)):
    """
    Builds the DLR-CovNet architecture.
    """

    inputs = layers.Input(shape=input_shape)

    # ------------------------------------------------------------------
    # Stem
    # ------------------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    branch_3x3 = rdcp_block(
        x,
        filters=32,
        kernel_size=3,
    )

    branch_5x5 = rdcp_block(
        x,
        filters=32,
        kernel_size=5,
    )

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    b11 = rdcp_block(branch_3x3, 64, 3)
    b12 = rdcp_block(branch_3x3, 64, 5)

    b21 = rdcp_block(branch_5x5, 64, 3)
    b22 = rdcp_block(branch_5x5, 64, 5)

    # ------------------------------------------------------------------
    # Cross-scale Feature Fusion
    # ------------------------------------------------------------------

    fusion_3x3 = layers.Concatenate()([b11, b21])
    fusion_5x5 = layers.Concatenate()([b12, b22])

    fusion_3x3 = rdcp_block(fusion_3x3, 128, 3)
    fusion_5x5 = rdcp_block(fusion_5x5, 128, 5)

    x = layers.Concatenate()([
        fusion_3x3,
        fusion_5x5,
    ])

    # ------------------------------------------------------------------
    # Classification Head
    # ------------------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    return models.Model(
        inputs=inputs,
        outputs=outputs,
        name="DLR-CovNet",
    )