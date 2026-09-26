"""Evaluate the untouched balanced test split, with melanoma as positive."""
import json
from pathlib import Path

from train import (ROOT, METRICS, FIGURES, setup, dataset, load_class_names,
                   load_model, melanoma_probabilities)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix, roc_curve)


def main():
    setup()
    model = load_model()
    names = load_class_names()
    test = dataset("test", names)
    if list(test.class_names) != names:
        raise SystemExit("Test class order differs from training.")
    files = list(test.file_paths)
    labels, probabilities = [], []
    for images, batch_labels in test:
        labels.extend(batch_labels.numpy().reshape(-1).astype(int).tolist())
        probabilities.extend(melanoma_probabilities(model(images, training=False).numpy(), names).tolist())
    probabilities = np.asarray(probabilities)
    truth = (np.asarray(labels) == names.index("melanoma")).astype(int)
    predicted = (probabilities >= 0.50).astype(int)
    binary_names = np.asarray(["non_melanoma", "melanoma"])
    pd.DataFrame({"filename": [Path(p).relative_to(ROOT).as_posix() for p in files],
                  "true_label": binary_names[truth], "predicted_label": binary_names[predicted],
                  "melanoma_probability": probabilities, "correct": truth == predicted}).to_csv(
                      ROOT / "outputs/predictions/test_predictions.csv", index=False)
    metrics = {"accuracy": float(accuracy_score(truth, predicted)),
               "precision": float(precision_score(truth, predicted, zero_division=0)),
               "recall": float(recall_score(truth, predicted, zero_division=0)),
               "f1_score": float(f1_score(truth, predicted, zero_division=0)),
               "roc_auc": float(roc_auc_score(truth, probabilities)) if len(np.unique(truth)) == 2 else None,
               "positive_class": "melanoma", "threshold": 0.50, "test_images": len(truth),
               "evaluation": "held-out balanced image-level test split"}
    (METRICS / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(truth, predicted, labels=[0, 1], target_names=binary_names, zero_division=0, digits=4)
    (METRICS / "classification_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion_matrix(truth, predicted, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                xticklabels=binary_names, yticklabels=binary_names, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Test confusion matrix (threshold 0.50)")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(truth, probabilities)
        ax.plot(fpr, tpr, label=f"Melanoma AUC = {metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC-AUC unavailable: only one test class", ha="center")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out test ROC", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curve.png", dpi=160)
    plt.close(fig)
    print(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
