"""Utilities for standardising media inputs and centring the response."""

from __future__ import annotations

import numpy as np


def _stat_vec(U: np.ndarray, stat: str) -> np.ndarray:
    """Return a per-column scale statistic for the given matrix."""

    if stat == "median":
        values = np.median(U, axis=0)
    elif stat == "mean":
        values = np.mean(U, axis=0)
    elif stat == "p95":
        values = np.percentile(U, 95, axis=0)
    else:
        raise ValueError("media_scale_stat must be 'median', 'mean', or 'p95'")

    values = np.asarray(values, dtype=float)
    # Avoid division by extremely small numbers.
    values = np.where(values <= 1e-12, 1.0, values)
    return values


def fit_media_scales_pre_adstock_tactical(
    U_tactical: np.ndarray, *, stat: str = "median"
) -> np.ndarray:
    """Return per-tactical positive scale factors using raw pre-adstock inputs."""

    return _stat_vec(U_tactical, stat)


def apply_pre_adstock_tactical(U_tactical: np.ndarray, s_t: np.ndarray) -> np.ndarray:
    """Scale each tactical series by the provided positive scale vector."""

    if s_t.ndim != 1 or s_t.shape[0] != U_tactical.shape[1]:
        raise ValueError("s_t must be a 1-D vector matching the tactical dimension")
    return U_tactical / s_t[None, :]


def center_y(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return centred response values and the mean that was removed."""

    mean = float(np.mean(y))
    return y - mean, mean
