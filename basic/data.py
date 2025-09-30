"""Data loading utilities for the basic machine learning pipeline."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the classic Iris dataset as Pandas objects.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, list[str]]
        Feature matrix, target vector, and the list of target names.
    """

    dataset = load_iris(as_frame=True)
    features: pd.DataFrame = dataset.data
    target: pd.Series = dataset.target
    target_names: list[str] = list(dataset.target_names)
    return features, target, target_names


__all__ = ["load_iris_data"]
