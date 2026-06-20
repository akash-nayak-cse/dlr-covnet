# model.py
# Multi-branch CNN with residual skip connections for binary classification

import tensorflow as tf
from tensorflow.keras import layers, models


def branch(input_tensor, filters, kernel):
    """
    Single branch block: two conv layers with a 1x1 skip connection,
    followed by BatchNorm and MaxPooling.

    Args:
        input_tensor : incoming feature map
        filters (int): number of conv filters
        kernel  (int): spatial kernel size (3 or 5)

    Returns:
        Tensor after pooling
    """
    skip = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(input_tensor)
    x    = layers.Conv2D(filters, (kernel, kernel), padding="same", activation="relu")(input_tensor)
    x    = layers.Conv2D(filters, (kernel, kernel), padding="same", activation="relu")(x)
    x    = layers.Add()([x, skip])
    x    = layers.BatchNormalization()(x)
    x    = layers.MaxPooling2D((2, 2))(x)
    return x


def create_model(input_shape=(224, 224, 1)):
    """
    Builds a two-level multi-branch CNN.

    Level 1 : two parallel branches (kernel 3 & 5) from the stem.
    Level 2 : four branches (3/5 from each level-1 branch), cross-concatenated
              into two streams, then merged at the final conv block.

    Returns:
        tf.keras.Model (binary output with sigmoid activation)
    """
    inputs = tf.keras.Input(shape=input_shape)

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ── Level 1: parallel branches ────────────────────────────────────────────
    b1 = branch(x, filters=32, kernel=3)   # 3×3 path
    b2 = branch(x, filters=32, kernel=5)   # 5×5 path

    # ── Level 2: four branches from level-1 outputs ───────────────────────────
    b11 = branch(b1, filters=64, kernel=3)
    b12 = branch(b1, filters=64, kernel=5)
    b21 = branch(b2, filters=64, kernel=3)
    b22 = branch(b2, filters=64, kernel=5)

    # Cross-concatenate (3×3 streams together, 5×5 streams together)
    concat1  = layers.Concatenate()([b11, b21])
    concat2  = layers.Concatenate()([b12, b22])
    last_b1  = branch(concat1, filters=128, kernel=3)
    last_b2  = branch(concat2, filters=128, kernel=5)
    concat   = layers.Concatenate()([last_b1, last_b2])

    # ── Head ──────────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(concat)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="MultiBranchCNN")
