"""
Evaluation utilities for DLR-CovNet.

This module contains helper functions used for evaluating model
predictions using commonly adopted classification metrics.

The complete evaluation pipeline accompanying the published work
is intentionally omitted from this repository.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
)


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"],
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "classification_report": report,
        "mcc": mcc,
    }


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot the confusion matrix.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    """

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.close()

    return roc_auc
