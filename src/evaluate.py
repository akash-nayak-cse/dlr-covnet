# evaluate.py
# Load trained model and report all metrics on the test set

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
)

import config
from data_loader import create_data_mapping, custom_generator


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved → {save_path}")


def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved → {save_path}")
    return roc_auc


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Load test data ────────────────────────────────────────────────────────
    test_labels = create_data_mapping(config.TEST_DIR, config.TEST_TXT)
    test_steps  = len(test_labels) // config.BATCH_SIZE
    test_gen    = custom_generator(config.TEST_DIR, test_labels,
                                   config.BATCH_SIZE, shuffle=False)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {config.FINAL_MODEL} ...")
    model = tf.keras.models.load_model(config.FINAL_MODEL)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_scores = model.predict(test_gen, steps=test_steps, verbose=1).flatten()
    y_pred   = (y_scores > 0.5).astype(int)
    y_true   = np.array(list(test_labels.values()))[: len(y_pred)]

    # ── Metrics ───────────────────────────────────────────────────────────────
    report = classification_report(y_true, y_pred,
                                   target_names=["Negative", "Positive"])
    mcc    = matthews_corrcoef(y_true, y_pred)
    roc_auc = plot_roc_curve(
        y_true, y_scores,
        os.path.join(config.OUTPUT_DIR, "roc_curve.png"),
    )

    print("\nClassification Report:\n", report)
    print(f"MCC  : {mcc:.4f}")
    print(f"AUC  : {roc_auc:.4f}")

    with open(os.path.join(config.OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nMCC : {mcc:.4f}\nAUC : {roc_auc:.4f}\n")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    np.save(os.path.join(config.OUTPUT_DIR, "confusion_matrix.npy"), cm)


if __name__ == "__main__":
    main()
