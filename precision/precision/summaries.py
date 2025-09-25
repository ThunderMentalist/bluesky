"""Posterior summaries and contribution calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .adstock import adstock_geometric_np
from .hierarchy import Hierarchy
from .sampling import PosteriorSamples


@dataclass
class Contributions:
    """Container for detailed channel, platform, and tactical contributions."""

    tactical: pd.DataFrame
    platform: pd.DataFrame
    channel: pd.DataFrame
    controls: pd.DataFrame
    intercept: pd.Series
    fitted: pd.Series
    tactical_totals: pd.Series
    platform_totals: pd.Series
    channel_totals: pd.Series
    controls_totals: pd.Series
    intercept_total: float
    fitted_total: float


def posterior_mean(samples: PosteriorSamples) -> dict[str, np.ndarray]:
    """Compute posterior means for each parameter."""

    def _mean(array: np.ndarray) -> np.ndarray:
        if array.size == 0:
            return array
        return np.mean(array, axis=(0, 1))

    return {
        "beta0": _mean(samples.beta0),
        "beta_channel": _mean(samples.beta_channel),
        "gamma": _mean(samples.gamma),
        "delta": _mean(samples.delta),
        "sigma": _mean(samples.sigma),
    }


def summarise_decay_rates(
    samples: PosteriorSamples,
    hierarchy: Hierarchy,
    *,
    ci: Tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """Return decay posterior summaries as a tidy DataFrame."""

    flat = samples.delta.reshape(-1, samples.delta.shape[-1])
    lower, upper = np.quantile(flat, [ci[0], ci[1]], axis=0)

    return pd.DataFrame(
        {
            "tactical": hierarchy.tactical_names,
            "decay_mean": flat.mean(axis=0),
            "decay_sd": flat.std(axis=0, ddof=1),
            f"decay_p{int(100 * ci[0])}": lower,
            f"decay_p{int(100 * ci[1])}": upper,
        }
    )


def compute_contributions_from_params(
    y: np.ndarray,
    U_tactical: np.ndarray,
    Z_controls: Optional[np.ndarray],
    control_names: Optional[List[str]],
    hierarchy: Hierarchy,
    *,
    beta0: np.ndarray,
    beta_channel: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    normalize_adstock: bool = False,
    time_index: Optional[pd.Index] = None,
) -> Contributions:
    """Compute contributions for a given parameter realisation."""

    T, num_tacticals = U_tactical.shape
    num_controls = 0 if Z_controls is None else Z_controls.shape[1]

    if time_index is None:
        time_index = pd.RangeIndex(T, name="time")

    adstocked = adstock_geometric_np(U_tactical, delta, normalize=normalize_adstock)
    beta_per_tactical = hierarchy.M_tc @ beta_channel
    tactical_contrib = adstocked * beta_per_tactical[None, :]
    platform_contrib = tactical_contrib @ hierarchy.M_tp
    channel_contrib = tactical_contrib @ hierarchy.M_tc

    if num_controls == 0 or Z_controls is None or gamma.size == 0:
        control_contrib = np.zeros((T, 0))
        control_names = control_names or []
    else:
        gamma = np.atleast_1d(gamma)
        if gamma.ndim == 1:
            control_contrib = Z_controls * gamma[None, :]
        else:
            control_contrib = Z_controls @ gamma
        if control_names is None:
            control_names = [f"control_{idx}" for idx in range(control_contrib.shape[1])]

    intercept_contrib = np.full(T, float(beta0))
    fitted = intercept_contrib + channel_contrib.sum(axis=1)
    if control_contrib.size > 0:
        fitted = fitted + control_contrib.sum(axis=1)

    tactical_df = pd.DataFrame(tactical_contrib, index=time_index, columns=hierarchy.tactical_names)
    platform_df = pd.DataFrame(platform_contrib, index=time_index, columns=hierarchy.platform_names)
    channel_df = pd.DataFrame(channel_contrib, index=time_index, columns=hierarchy.channel_names)
    controls_df = pd.DataFrame(control_contrib, index=time_index, columns=control_names)
    intercept_series = pd.Series(intercept_contrib, index=time_index, name="intercept")
    fitted_series = pd.Series(fitted, index=time_index, name="fitted")

    return Contributions(
        tactical=tactical_df,
        platform=platform_df,
        channel=channel_df,
        controls=controls_df,
        intercept=intercept_series,
        fitted=fitted_series,
        tactical_totals=tactical_df.sum(axis=0),
        platform_totals=platform_df.sum(axis=0),
        channel_totals=channel_df.sum(axis=0),
        controls_totals=controls_df.sum(axis=0),
        intercept_total=float(intercept_series.sum()),
        fitted_total=float(fitted_series.sum()),
    )


__all__ = [
    "Contributions",
    "posterior_mean",
    "summarise_decay_rates",
    "compute_contributions_from_params",
]
