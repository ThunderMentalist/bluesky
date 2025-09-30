"""Demo script that runs the full decision tree training pipeline."""

from __future__ import annotations

from pathlib import Path

from .data import load_iris_data
from .evaluation import evaluate_model, plot_accuracy_progression
from .pipeline import create_dataset_splits, perform_hyperparameter_search, train_final_model


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_demo() -> None:
    """Execute the training pipeline and report metrics."""

    features, target, _ = load_iris_data()
    splits = create_dataset_splits(features, target)

    estimator, best_params = perform_hyperparameter_search(splits)
    print("Best hyper-parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    tuned_metrics = evaluate_model(estimator, splits)
    print("\nAccuracy after hyper-parameter tuning:")
    for split_name, data in tuned_metrics.items():
        print(f"  {split_name.capitalize()}: {data['accuracy']:.3f}")

    print("\nValidation classification report:")
    print(tuned_metrics["validation"]["classification_report"])

    final_estimator = train_final_model(estimator, splits)
    final_metrics = evaluate_model(final_estimator, splits)

    print("\nFinal model accuracy (retrained on train + validation):")
    for split_name, data in final_metrics.items():
        print(f"  {split_name.capitalize()}: {data['accuracy']:.3f}")

    chart_path = plot_accuracy_progression(tuned_metrics, OUTPUT_DIR / "accuracy_progression.png")
    print(f"\nAccuracy chart saved to: {chart_path}")


if __name__ == "__main__":
    run_demo()
