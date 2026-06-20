# train.py
# Entry point: builds, trains, and saves the model

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config
from model       import create_model
from data_loader import create_data_mapping, calculate_class_weights, custom_generator


def save_model_summary(model):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.SUMMARY_PATH, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"],     label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "training_history.png"), dpi=150)
    plt.show()


def main():
    # Reproducibility
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data mappings...")
    train_labels = create_data_mapping(config.TRAIN_DIR, config.TRAIN_TXT)
    val_labels   = create_data_mapping(config.VAL_DIR,   config.VAL_TXT)

    dist = np.bincount(list(train_labels.values()))
    print(f"  Train  — Negative: {dist[0]}  Positive: {dist[1]}")

    class_weights = calculate_class_weights(train_labels)
    print(f"  Class weights: {class_weights}")

    # ── Generators ────────────────────────────────────────────────────────────
    train_gen = custom_generator(config.TRAIN_DIR, train_labels,
                                 config.BATCH_SIZE, class_weights=class_weights)
    val_gen   = custom_generator(config.VAL_DIR,   val_labels,
                                 config.BATCH_SIZE, shuffle=False)

    train_steps = len(train_labels) // config.BATCH_SIZE
    val_steps   = len(val_labels)   // config.BATCH_SIZE

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model...")
    model = create_model(config.INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    save_model_summary(model)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            config.WEIGHTS_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    print("Training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history)
    np.save(os.path.join(config.OUTPUT_DIR, "training_history.npy"), history.history)

    model.save(config.FINAL_MODEL)
    print(f"Model saved → {config.FINAL_MODEL}")


if __name__ == "__main__":
    main()
