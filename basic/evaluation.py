"""Evaluation and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report

from .pipeline import DatasetSplits


def evaluate_model(
    estimator: ClassifierMixin,
    splits: DatasetSplits,
) -> Dict[str, Dict[str, float | str]]:
    """Compute classification metrics for train, validation, and test datasets."""

    metrics: Dict[str, Dict[str, float | str]] = {}
    for split_name, X, y in [
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ]:
        predictions = estimator.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(
            y,
            predictions,
            output_dict=False,
            zero_division=0,
        )
        metrics[split_name] = {
            "accuracy": accuracy,
            "classification_report": report,
        }

    return metrics


def plot_accuracy_progression(
    metrics: Dict[str, Dict[str, float | str]],
    output_path: Path,
) -> Path:
    """Create a simple bar chart of accuracy across dataset splits."""

    accuracies = {
        split_name: data["accuracy"]
        for split_name, data in metrics.items()
        if "accuracy" in data
    }
    ordered_splits = ["train", "validation", "test"]
    y_values = [accuracies.get(name, 0.0) for name in ordered_splits]

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(ordered_splits, y_values, color=["#4c72b0", "#55a868", "#c44e52"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Decision Tree Accuracy Progression")
    axis.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, value in zip(bars, y_values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


__all__ = ["evaluate_model", "plot_accuracy_progression"]
