"""Utilities for constructing designed variation control matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fourier_seasonality(
    length: int,
    *,
    period: int = 52,
    harmonics: int = 3,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of Fourier seasonal harmonics.

    Parameters
    ----------
    length:
        Number of rows (typically the number of time periods).
    period:
        Seasonality period (e.g. ``52`` for weekly data with yearly seasonality).
    harmonics:
        Number of sine/cosine pairs to include.
    index:
        Optional index to attach to the returned DataFrame.
    """

    t = np.arange(length)
    data: dict[str, np.ndarray] = {}
    for k in range(1, harmonics + 1):
        angle = 2.0 * np.pi * k * t / period
        data[f"sin_{period}_{k}"] = np.sin(angle)
        data[f"cos_{period}_{k}"] = np.cos(angle)
    idx = index if index is not None else pd.RangeIndex(length, name="time")
    return pd.DataFrame(data, index=idx)


def flag_weeks(
    length: int,
    flags: dict[str, list[int]],
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Construct indicator columns for holiday, promotion, or custom events."""

    idx = index if index is not None else pd.RangeIndex(length, name="time")
    frame = pd.DataFrame(0, index=idx, columns=sorted(flags.keys()), dtype=float)
    for name, weeks in flags.items():
        valid = [week for week in weeks if 0 <= week < length]
        frame.loc[idx[valid], name] = 1.0
    return frame


def stack_controls(*frames: pd.DataFrame) -> np.ndarray:
    """Column-stack multiple control DataFrames into a design matrix."""

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return np.zeros((0, 0), dtype=float)
    base_index = frames[0].index
    aligned = [frame.reindex(base_index).fillna(0.0) for frame in frames]
    return np.column_stack([frame.to_numpy() for frame in aligned])


__all__ = ["fourier_seasonality", "flag_weeks", "stack_controls"]
