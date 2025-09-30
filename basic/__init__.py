"""Basic machine learning pipeline package."""

from .data import load_iris_data
from .pipeline import (  # noqa: F401
    DatasetSplits,
    create_dataset_splits,
    perform_hyperparameter_search,
    train_final_model,
)
from .evaluation import evaluate_model, plot_accuracy_progression  # noqa: F401

__all__ = [
    "load_iris_data",
    "DatasetSplits",
    "create_dataset_splits",
    "perform_hyperparameter_search",
    "train_final_model",
    "evaluate_model",
    "plot_accuracy_progression",
]
