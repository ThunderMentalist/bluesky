"""Utilities for constructing designed variation control matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fourier_seasonality(
    T: int,
    *,
    period: int = 52,
    K: int = 3,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of sine/cosine Fourier harmonics."""

    t = np.arange(T)
    cols: dict[str, np.ndarray] = {}
    for k in range(1, K + 1):
        angle = 2.0 * np.pi * k * t / period
        cols[f"sin_{period}_{k}"] = np.sin(angle)
        cols[f"cos_{period}_{k}"] = np.cos(angle)
    idx = index if index is not None else pd.RangeIndex(T, name="time")
    return pd.DataFrame(cols, index=idx)


def holiday_flags(index: pd.Index, spans: dict[str, list[int]]) -> pd.DataFrame:
    """Construct indicator columns for holiday, promotion, or custom spans."""

    frame = pd.DataFrame(0.0, index=index, columns=sorted(spans.keys()))
    n = len(index)
    for name, weeks in spans.items():
        valid = [w for w in weeks if 0 <= w < n]
        if valid:
            frame.loc[index[valid], name] = 1.0
    return frame


def stack_controls(*frames: pd.DataFrame, standardize: bool = True) -> np.ndarray:
    """Column-stack aligned frames and optionally z-score columns."""

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return np.zeros((0, 0), dtype=float)
    base = frames[0].index
    aligned = [frame.reindex(base).fillna(0.0) for frame in frames]
    X = np.column_stack([a.to_numpy() for a in aligned])
    if standardize and X.size:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=0, keepdims=True)
        sd = np.where(sd <= 1e-9, 1.0, sd)
        X = (X - mu) / sd
    return X


__all__ = ["fourier_seasonality", "holiday_flags", "stack_controls"]
