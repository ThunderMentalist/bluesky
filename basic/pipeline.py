"""Core training utilities for the decision tree pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DatasetSplits:
    """Container for train, validation, and test splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


SplitReturn = Tuple[pd.DataFrame, pd.Series]


def create_dataset_splits(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """Create train, validation, and test splits from the dataset."""

    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    validation_ratio = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def perform_hyperparameter_search(
    splits: DatasetSplits,
    *,
    cv_folds: int = 5,
    scoring: str = "accuracy",
) -> Tuple[ClassifierMixin, Dict[str, Any]]:
    """Run grid search cross-validation to optimise model hyper-parameters."""

    param_grid: Dict[str, list[Any]] = {
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3],
        "criterion": ["gini", "entropy"],
    }

    search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
    )
    search.fit(splits.X_train, splits.y_train)
    return search.best_estimator_, search.best_params_


def train_final_model(
    estimator: ClassifierMixin,
    splits: DatasetSplits,
) -> ClassifierMixin:
    """Retrain the estimator on the combined train and validation sets."""

    X_combined = pd.concat([splits.X_train, splits.X_val], axis=0)
    y_combined = pd.concat([splits.y_train, splits.y_val], axis=0)
    estimator.fit(X_combined, y_combined)
    return estimator


__all__ = [
    "DatasetSplits",
    "create_dataset_splits",
    "perform_hyperparameter_search",
    "train_final_model",
]
