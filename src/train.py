"""
Training utilities for DLR-CovNet.

This module outlines the training configuration used for the
published DLR-CovNet architecture. 

The complete training
pipeline accompanying the Springer publication has been
intentionally omitted from this repository.
"""

import tensorflow as tf

import config
from model import create_model


def build_compiled_model():
    """
    Builds and compiles the DLR-CovNet model.
    """

    model = create_model(config.INPUT_SHAPE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks():
    """
    Returns the callbacks used during training.
    """

    return [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
        ),

    ]