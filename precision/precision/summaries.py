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
class ContributionInterval:
    lower: pd.DataFrame
    upper: pd.DataFrame


@dataclass
class ContributionIntervalSeries:
    lower: pd.Series
    upper: pd.Series


@dataclass
class ContributionUncertainty:
    tactical: ContributionInterval
    platform: ContributionInterval
    channel: ContributionInterval
    controls: ContributionInterval
    intercept: ContributionIntervalSeries
    fitted: ContributionIntervalSeries


def compute_contribution_arrays(
    U: np.ndarray,
    Z: Optional[np.ndarray],
    hierarchy: Hierarchy,
    *,
    beta0: float,
    beta_channel: Optional[np.ndarray],
    gamma: Optional[np.ndarray],
    delta: np.ndarray,
    normalize_adstock: bool,
    beta_platform: Optional[np.ndarray] = None,
    beta_tactical: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-draw contribution arrays (tactical/platform/channel/controls/intercept/fitted)."""

    adstocked = adstock_geometric_np(U, delta, normalize=normalize_adstock)
    beta_per_tactical = beta_per_tactical_from_params(
        hierarchy,
        beta_channel=beta_channel,
        beta_platform=beta_platform,
        beta_tactical=beta_tactical,
    )
    tactical = adstocked * beta_per_tactical[None, :]
    platform = tactical @ hierarchy.M_tp
    channel = tactical @ hierarchy.M_tc

    if Z is None or gamma is None or gamma.size == 0:
        control = np.zeros((U.shape[0], 0))
    else:
        gamma = np.atleast_1d(gamma)
        control = (Z * gamma[None, :]) if gamma.ndim == 1 else (Z @ gamma)

    intercept = np.full(U.shape[0], float(beta0))
    fitted = intercept + channel.sum(axis=1) + (control.sum(axis=1) if control.size > 0 else 0.0)
    return tactical, platform, channel, control, intercept, fitted


def _to_df(values: np.ndarray, index: pd.Index, columns: list[str]) -> pd.DataFrame:
    if values.size == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    return pd.DataFrame(values, index=index, columns=columns)


def _to_series(values: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    return pd.Series(values, index=index, name=name)


def _summarise_draws(
    draws: np.ndarray,
    index: pd.Index,
    columns: list[str],
    *,
    lower: float = 0.05,
    upper: float = 0.95,
) -> ContributionInterval:
    if draws.size == 0:
        empty = pd.DataFrame(index=index, columns=columns, dtype=float)
        return ContributionInterval(lower=empty.copy(), upper=empty.copy())
    lo = np.quantile(draws, lower, axis=0)
    hi = np.quantile(draws, upper, axis=0)
    return ContributionInterval(
        lower=pd.DataFrame(lo, index=index, columns=columns),
        upper=pd.DataFrame(hi, index=index, columns=columns),
    )


def _summarise_draws_series(
    draws: np.ndarray,
    index: pd.Index,
    name: str,
    *,
    lower: float = 0.05,
    upper: float = 0.95,
) -> ContributionIntervalSeries:
    lo = np.quantile(draws, lower, axis=0)
    hi = np.quantile(draws, upper, axis=0)
    return ContributionIntervalSeries(
        lower=pd.Series(lo, index=index, name=name),
        upper=pd.Series(hi, index=index, name=name),
    )


@dataclass
class ContributionDraws:
    """Holds per-draw arrays for downstream uncertainty summaries."""

    tactical: np.ndarray
    platform: np.ndarray
    channel: np.ndarray
    controls: np.ndarray
    intercept: np.ndarray
    fitted: np.ndarray


def contributions_from_posterior(
    samples: PosteriorSamples,
    *,
    y: np.ndarray,
    U_tactical: np.ndarray,
    Z_controls: Optional[np.ndarray],
    hierarchy: Hierarchy,
    control_names: Optional[list[str]] = None,
    normalize_adstock: bool = False,
    ci: Tuple[float, float] = (0.05, 0.95),
    return_draws: bool = False,
) -> Tuple[Contributions, ContributionUncertainty, Optional[ContributionDraws]]:
    """Compute per-draw contributions and summary uncertainty statistics."""

    stacked = samples.stack_chains()
    D = stacked.beta0.shape[0]
    T, N_t = U_tactical.shape
    P = hierarchy.num_platforms
    C = hierarchy.num_channels
    J = 0 if Z_controls is None else Z_controls.shape[1]

    index = pd.RangeIndex(T, name="time")
    ctrl_names = control_names or [f"control_{j}" for j in range(J)]

    tactical_draws = np.zeros((D, T, N_t))
    platform_draws = np.zeros((D, T, P))
    channel_draws = np.zeros((D, T, C))
    controls_draws = np.zeros((D, T, J))
    intercept_draws = np.zeros((D, T))
    fitted_draws = np.zeros((D, T))

    for d in range(D):
        beta_channel = stacked.beta_channel[d] if stacked.beta_channel.size > 0 else None
        beta_platform = None if stacked.beta_platform is None else stacked.beta_platform[d]
        beta_tactical = None if stacked.beta_tactical is None else stacked.beta_tactical[d]
        gamma_d = None
        if stacked.gamma is not None and stacked.gamma.size > 0:
            gamma_d = stacked.gamma[d]
        arrays = compute_contribution_arrays(
            U_tactical,
            Z_controls,
            hierarchy,
            beta0=float(stacked.beta0[d]),
            beta_channel=beta_channel,
            gamma=gamma_d,
            delta=stacked.delta[d],
            normalize_adstock=normalize_adstock,
            beta_platform=beta_platform,
            beta_tactical=beta_tactical,
        )
        tactical, platform, channel, controls_arr, intercept, fitted = arrays
        tactical_draws[d] = tactical
        platform_draws[d] = platform
        channel_draws[d] = channel
        if J > 0:
            controls_draws[d] = controls_arr
        intercept_draws[d] = intercept
        fitted_draws[d] = fitted

    mean_tactical = tactical_draws.mean(axis=0)
    mean_platform = platform_draws.mean(axis=0)
    mean_channel = channel_draws.mean(axis=0)
    mean_controls = controls_draws.mean(axis=0) if J > 0 else controls_draws.mean(axis=0)
    mean_intercept = intercept_draws.mean(axis=0)
    mean_fitted = fitted_draws.mean(axis=0)

    contributions = Contributions(
        tactical=_to_df(mean_tactical, index, hierarchy.tactical_names),
        platform=_to_df(mean_platform, index, hierarchy.platform_names),
        channel=_to_df(mean_channel, index, hierarchy.channel_names),
        controls=_to_df(mean_controls, index, ctrl_names),
        intercept=_to_series(mean_intercept, index, "intercept"),
        fitted=_to_series(mean_fitted, index, "fitted"),
        tactical_totals=pd.Series(mean_tactical.sum(axis=0), index=hierarchy.tactical_names),
        platform_totals=pd.Series(mean_platform.sum(axis=0), index=hierarchy.platform_names),
        channel_totals=pd.Series(mean_channel.sum(axis=0), index=hierarchy.channel_names),
        controls_totals=(
            pd.Series(mean_controls.sum(axis=0), index=ctrl_names)
            if J > 0
            else pd.Series(dtype=float, index=pd.Index(ctrl_names))
        ),
        intercept_total=float(mean_intercept.sum()),
        fitted_total=float(mean_fitted.sum()),
    )

    lower, upper = ci
    uncertainty = ContributionUncertainty(
        tactical=_summarise_draws(tactical_draws, index, hierarchy.tactical_names, lower=lower, upper=upper),
        platform=_summarise_draws(platform_draws, index, hierarchy.platform_names, lower=lower, upper=upper),
        channel=_summarise_draws(channel_draws, index, hierarchy.channel_names, lower=lower, upper=upper),
        controls=_summarise_draws(controls_draws, index, ctrl_names, lower=lower, upper=upper),
        intercept=_summarise_draws_series(intercept_draws, index, "intercept", lower=lower, upper=upper),
        fitted=_summarise_draws_series(fitted_draws, index, "fitted", lower=lower, upper=upper),
    )

    draws = (
        ContributionDraws(
            tactical=tactical_draws,
            platform=platform_draws,
            channel=channel_draws,
            controls=controls_draws,
            intercept=intercept_draws,
            fitted=fitted_draws,
        )
        if return_draws
        else None
    )

    return contributions, uncertainty, draws


@dataclass
class ContributionSignProb:
    tactical_ts: pd.DataFrame
    platform_ts: pd.DataFrame
    channel_ts: pd.DataFrame
    tactical_total: pd.Series
    platform_total: pd.Series
    channel_total: pd.Series


def contribution_sign_probabilities(
    draws: ContributionDraws,
    *,
    hierarchy: Hierarchy,
) -> ContributionSignProb:
    """Probability that contributions are positive for each draw."""

    T = draws.tactical.shape[1]
    index = pd.RangeIndex(T, name="time")

    tactical_prob_ts = (draws.tactical > 0).mean(axis=0)
    platform_prob_ts = (draws.platform > 0).mean(axis=0)
    channel_prob_ts = (draws.channel > 0).mean(axis=0)

    tactical_total_prob = (draws.tactical.sum(axis=1) > 0).mean(axis=0)
    platform_total_prob = (draws.platform.sum(axis=1) > 0).mean(axis=0)
    channel_total_prob = (draws.channel.sum(axis=1) > 0).mean(axis=0)

    return ContributionSignProb(
        tactical_ts=pd.DataFrame(tactical_prob_ts, index=index, columns=hierarchy.tactical_names),
        platform_ts=pd.DataFrame(platform_prob_ts, index=index, columns=hierarchy.platform_names),
        channel_ts=pd.DataFrame(channel_prob_ts, index=index, columns=hierarchy.channel_names),
        tactical_total=pd.Series(tactical_total_prob, index=hierarchy.tactical_names, name="Pr>0"),
        platform_total=pd.Series(platform_total_prob, index=hierarchy.platform_names, name="Pr>0"),
        channel_total=pd.Series(channel_total_prob, index=hierarchy.channel_names, name="Pr>0"),
    )


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


def posterior_mean(samples: PosteriorSamples) -> dict[str, np.ndarray | None]:
    """Compute posterior means for each parameter."""

    def _mean(array: np.ndarray | None) -> np.ndarray | None:
        if array is None:
            return None
        if array.size == 0:
            return array
        return np.mean(array, axis=(0, 1))

    return {
        "beta0": _mean(samples.beta0),
        "beta_channel": _mean(samples.beta_channel),
        "gamma": _mean(samples.gamma),
        "delta": _mean(samples.delta),
        "sigma": _mean(samples.sigma),
        "beta_platform": _mean(samples.beta_platform),
        "beta_tactical": _mean(samples.beta_tactical),
        "tau_beta": _mean(samples.tau_beta),
        "tau0": _mean(samples.tau0),
        "lambda_local": _mean(samples.lambda_local),
        "s_sat": _mean(samples.s_sat),
    }


def _saturate_log1p_np(x: np.ndarray, s: float) -> np.ndarray:
    """Apply the minor saturation transform elementwise."""

    s_safe = float(max(s, 1e-9))
    return s_safe * np.log1p(x / s_safe)


def beta_per_tactical_from_params(
    hierarchy: Hierarchy,
    *,
    beta_channel: Optional[np.ndarray] = None,
    beta_platform: Optional[np.ndarray] = None,
    beta_tactical: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return per-tactical coefficients regardless of modelling structure."""

    if beta_tactical is not None and beta_tactical.size > 0:
        return np.asarray(beta_tactical, dtype=float)
    if beta_platform is not None and beta_platform.size > 0:
        return hierarchy.M_tp @ np.asarray(beta_platform, dtype=float)
    if beta_channel is not None and beta_channel.size > 0:
        return hierarchy.M_tc @ np.asarray(beta_channel, dtype=float)
    raise ValueError(
        "At least one of beta_channel, beta_platform, or beta_tactical must be provided."
    )


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
    beta_channel: Optional[np.ndarray] = None,
    gamma: np.ndarray,
    delta: np.ndarray,
    beta_platform: Optional[np.ndarray] = None,
    beta_tactical: Optional[np.ndarray] = None,
    normalize_adstock: bool = True,
    time_index: Optional[pd.Index] = None,
    saturation_scale: Optional[float] = None,
) -> Contributions:
    """Compute contributions for a given parameter realisation.

    When ``normalize_adstock`` is ``True`` (default) the tactical signals are
    adstocked using the normalized variant to align with model fitting.
    """

    T, num_tacticals = U_tactical.shape
    num_controls = 0 if Z_controls is None else Z_controls.shape[1]

    if time_index is None:
        time_index = pd.RangeIndex(T, name="time")

    adstocked = adstock_geometric_np(U_tactical, delta, normalize=normalize_adstock)
    if saturation_scale is not None:
        if saturation_scale <= 0:
            raise ValueError("saturation_scale must be positive")
        adstocked = _saturate_log1p_np(adstocked, float(saturation_scale))
    beta_per_tactical = beta_per_tactical_from_params(
        hierarchy,
        beta_channel=beta_channel,
        beta_platform=beta_platform,
        beta_tactical=beta_tactical,
    )
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
    "ContributionInterval",
    "ContributionIntervalSeries",
    "ContributionUncertainty",
    "ContributionDraws",
    "ContributionSignProb",
    "Contributions",
    "compute_contribution_arrays",
    "contributions_from_posterior",
    "contribution_sign_probabilities",
    "posterior_mean",
    "summarise_decay_rates",
    "compute_contributions_from_params",
    "beta_per_tactical_from_params",
]
