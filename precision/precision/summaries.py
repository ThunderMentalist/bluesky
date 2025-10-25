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


def beta_per_leaf_from_params(
    *,
    beta: np.ndarray,
    beta_level: str,
    hierarchy: Hierarchy,
    leaf_level: str,
) -> np.ndarray:
    """Project coefficients from ``beta_level`` down to the ``leaf_level``."""

    coeff = np.asarray(beta, dtype=float)
    mapping = hierarchy.map(leaf_level, beta_level)
    return mapping @ coeff


def compute_contribution_arrays(
    *,
    U_raw: np.ndarray,
    delta: np.ndarray,
    beta: np.ndarray,
    beta_level: str,
    hierarchy: Hierarchy,
    leaf_level: str,
    normalize_adstock: bool,
    use_saturation: bool,
    s_sat: float | None,
    tactical_rescale: Optional[np.ndarray] = None,
    levels: Optional[List[str]] = None,
) -> dict[str, np.ndarray]:
    """Return contribution arrays for the leaf level and requested ancestors."""

    if tactical_rescale is None:
        U_scaled = U_raw
    else:
        scale = np.asarray(tactical_rescale, dtype=float)
        if scale.ndim == 0:
            factor = float(scale)
            U_scaled = U_raw / factor
        else:
            U_scaled = U_raw / scale[None, :]
    adstocked = adstock_geometric_np(U_scaled, delta, normalize=normalize_adstock)

    if use_saturation:
        s_val = max(float(s_sat or 0.0), 1e-6)
        adstocked = s_val * np.log1p(adstocked / s_val)

    beta_leaf = beta_per_leaf_from_params(
        beta=beta,
        beta_level=beta_level,
        hierarchy=hierarchy,
        leaf_level=leaf_level,
    )
    leaf_contrib = adstocked * beta_leaf[None, :]

    if levels is None:
        leaf_idx = hierarchy.levels.index(leaf_level)
        levels = hierarchy.levels[leaf_idx + 1 :]

    contributions: dict[str, np.ndarray] = {leaf_level: leaf_contrib}
    for level in levels:
        matrix = hierarchy.map(leaf_level, level)
        contributions[level] = leaf_contrib @ matrix
    return contributions


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
    beta0_offset: float = 0.0,
    tactical_scale: Optional[np.ndarray] = None,
    ci: Tuple[float, float] = (0.05, 0.95),
    return_draws: bool = False,
) -> Tuple[Contributions, ContributionUncertainty, Optional[ContributionDraws]]:
    """Compute per-draw contributions and summary uncertainty statistics."""

    stacked = samples.stack_chains()
    D = stacked.beta0.shape[0]
    T = U_tactical.shape[0]
    leaf_level = hierarchy.levels[0]
    leaf_names = hierarchy.names[leaf_level]
    N_t = len(leaf_names)

    platform_level = hierarchy.levels[1] if len(hierarchy.levels) > 1 else None
    platform_names = hierarchy.names[platform_level] if platform_level is not None else []
    P = len(platform_names)

    channel_level = hierarchy.levels[2] if len(hierarchy.levels) > 2 else None
    channel_names = hierarchy.names[channel_level] if channel_level is not None else []
    C = len(channel_names)
    J = 0 if Z_controls is None else Z_controls.shape[1]

    index = pd.RangeIndex(T, name="time")
    ctrl_names = control_names or [f"control_{j}" for j in range(J)]

    if U_tactical.shape[1] != N_t:
        raise ValueError(
            "U_tactical column count must match the number of leaf nodes in the hierarchy"
        )

    tactical_draws = np.zeros((D, T, N_t))
    platform_draws = np.zeros((D, T, P))
    channel_draws = np.zeros((D, T, C))
    controls_draws = np.zeros((D, T, J))
    intercept_draws = np.zeros((D, T))
    fitted_draws = np.zeros((D, T))

    has_saturation = stacked.s_sat is not None and stacked.s_sat.size > 0

    for d in range(D):
        beta_channel = stacked.beta_channel[d] if stacked.beta_channel.size > 0 else None
        beta_platform = None if stacked.beta_platform is None else stacked.beta_platform[d]
        beta_tactical = None if stacked.beta_tactical is None else stacked.beta_tactical[d]
        beta_leaf = beta_per_tactical_from_params(
            hierarchy,
            beta_channel=beta_channel,
            beta_platform=beta_platform,
            beta_tactical=beta_tactical,
            tactical_scale=tactical_scale,
        )

        if has_saturation:
            s_draw = np.ravel(stacked.s_sat[d])
            s_sat_draw = float(s_draw[0]) if s_draw.size > 0 else None
            use_saturation = s_sat_draw is not None
        else:
            s_sat_draw = None
            use_saturation = False

        contribs = compute_contribution_arrays(
            U_raw=U_tactical,
            delta=stacked.delta[d],
            beta=beta_leaf,
            beta_level=leaf_level,
            hierarchy=hierarchy,
            leaf_level=leaf_level,
            normalize_adstock=normalize_adstock,
            use_saturation=use_saturation,
            s_sat=s_sat_draw,
        )

        tactical = contribs[leaf_level]
        platform = (
            contribs.get(platform_level, np.zeros((T, P)))
            if platform_level is not None
            else np.zeros((T, P))
        )
        channel = (
            contribs.get(channel_level, np.zeros((T, C)))
            if channel_level is not None
            else np.zeros((T, C))
        )

        gamma_d = None
        if stacked.gamma is not None and stacked.gamma.size > 0:
            gamma_d = stacked.gamma[d]
        if Z_controls is None or gamma_d is None or gamma_d.size == 0:
            controls_arr = np.zeros((T, J))
        else:
            gamma_arr = np.atleast_1d(gamma_d)
            controls_arr = (
                Z_controls * gamma_arr[None, :]
                if gamma_arr.ndim == 1
                else Z_controls @ gamma_arr
            )

        intercept = np.full(T, float(stacked.beta0[d] + beta0_offset))
        top_level = hierarchy.levels[-1]
        marketing_total = (
            contribs[top_level].sum(axis=1)
            if top_level in contribs
            else tactical.sum(axis=1)
        )
        control_total = controls_arr.sum(axis=1) if controls_arr.size > 0 else 0.0
        fitted = intercept + marketing_total + control_total

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
        tactical=_to_df(mean_tactical, index, leaf_names),
        platform=_to_df(mean_platform, index, platform_names),
        channel=_to_df(mean_channel, index, channel_names),
        controls=_to_df(mean_controls, index, ctrl_names),
        intercept=_to_series(mean_intercept, index, "intercept"),
        fitted=_to_series(mean_fitted, index, "fitted"),
        tactical_totals=pd.Series(mean_tactical.sum(axis=0), index=leaf_names),
        platform_totals=pd.Series(mean_platform.sum(axis=0), index=platform_names),
        channel_totals=pd.Series(mean_channel.sum(axis=0), index=channel_names),
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
        tactical=_summarise_draws(tactical_draws, index, leaf_names, lower=lower, upper=upper),
        platform=_summarise_draws(platform_draws, index, platform_names, lower=lower, upper=upper),
        channel=_summarise_draws(channel_draws, index, channel_names, lower=lower, upper=upper),
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

    leaf_level = hierarchy.levels[0]
    leaf_names = hierarchy.names[leaf_level]
    platform_level = hierarchy.levels[1] if len(hierarchy.levels) > 1 else None
    platform_names = hierarchy.names[platform_level] if platform_level is not None else []
    channel_level = hierarchy.levels[2] if len(hierarchy.levels) > 2 else None
    channel_names = hierarchy.names[channel_level] if channel_level is not None else []

    return ContributionSignProb(
        tactical_ts=pd.DataFrame(tactical_prob_ts, index=index, columns=leaf_names),
        platform_ts=pd.DataFrame(platform_prob_ts, index=index, columns=platform_names),
        channel_ts=pd.DataFrame(channel_prob_ts, index=index, columns=channel_names),
        tactical_total=pd.Series(tactical_total_prob, index=leaf_names, name="Pr>0"),
        platform_total=pd.Series(platform_total_prob, index=platform_names, name="Pr>0"),
        channel_total=pd.Series(channel_total_prob, index=channel_names, name="Pr>0"),
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
        "eta_channel": _mean(samples.eta_channel),
        "eta_platform": _mean(samples.eta_platform),
        "eta_tactical": _mean(samples.eta_tactical),
        "eta_by_level": {
            level: _mean(array) for level, array in samples.eta_by_level.items()
        },
        "beta_by_level": {
            level: _mean(array) for level, array in samples.beta_by_level.items()
        },
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
    tactical_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return per-tactical coefficients regardless of modelling structure."""

    leaf_level = hierarchy.levels[0]
    leaf_idx = hierarchy.levels.index(leaf_level)

    def _map_to_leaf(
        coeff: np.ndarray,
        level_hint: Optional[str],
        start_level_idx: int,
        *,
        prefer_highest: bool,
    ) -> np.ndarray:
        size = coeff.size
        if level_hint is not None and level_hint in hierarchy.names:
            if hierarchy.size(level_hint) == size:
                return beta_per_leaf_from_params(
                    beta=coeff,
                    beta_level=level_hint,
                    hierarchy=hierarchy,
                    leaf_level=leaf_level,
                )

        search_levels = hierarchy.levels[start_level_idx:]
        iterable = reversed(search_levels) if prefer_highest else search_levels
        for level in iterable:
            if hierarchy.size(level) == size:
                return beta_per_leaf_from_params(
                    beta=coeff,
                    beta_level=level,
                    hierarchy=hierarchy,
                    leaf_level=leaf_level,
                )
        raise ValueError(
            "No hierarchy level matches provided beta size; ensure hierarchy levels are configured"
        )

    if beta_tactical is not None and beta_tactical.size > 0:
        beta = np.asarray(beta_tactical, dtype=float)
    elif beta_platform is not None and beta_platform.size > 0:
        coeff = np.asarray(beta_platform, dtype=float)
        beta = _map_to_leaf(
            coeff,
            "platform" if "platform" in hierarchy.names else None,
            leaf_idx + 1,
            prefer_highest=False,
        )
    elif beta_channel is not None and beta_channel.size > 0:
        coeff = np.asarray(beta_channel, dtype=float)
        beta = _map_to_leaf(
            coeff,
            "channel" if "channel" in hierarchy.names else None,
            leaf_idx + 1,
            prefer_highest=True,
        )
    else:
        raise ValueError(
            "At least one of beta_channel, beta_platform, or beta_tactical must be provided."
        )
    if tactical_scale is not None:
        scale = np.asarray(tactical_scale, dtype=float)
        if scale.size not in (0, beta.size):
            raise ValueError("tactical_scale must match tactical dimension or be None")
        if scale.size == beta.size:
            beta = beta * scale
    return beta


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
    beta0_offset: float = 0.0,
    tactical_scale: Optional[np.ndarray] = None,
) -> Contributions:
    """Compute contributions for a given parameter realisation.

    When ``normalize_adstock`` is ``True`` (default) the tactical signals are
    adstocked using the normalized variant to align with model fitting.
    """

    T = U_tactical.shape[0]
    num_controls = 0 if Z_controls is None else Z_controls.shape[1]

    if time_index is None:
        time_index = pd.RangeIndex(T, name="time")

    leaf_level = hierarchy.levels[0]
    leaf_names = hierarchy.names[leaf_level]
    if U_tactical.shape[1] != len(leaf_names):
        raise ValueError(
            "U_tactical column count must match the number of leaf nodes in the hierarchy"
        )

    platform_level = hierarchy.levels[1] if len(hierarchy.levels) > 1 else None
    platform_names = hierarchy.names[platform_level] if platform_level is not None else []
    channel_level = hierarchy.levels[2] if len(hierarchy.levels) > 2 else None
    channel_names = hierarchy.names[channel_level] if channel_level is not None else []

    beta_leaf = beta_per_tactical_from_params(
        hierarchy,
        beta_channel=beta_channel,
        beta_platform=beta_platform,
        beta_tactical=beta_tactical,
        tactical_scale=tactical_scale,
    )

    use_saturation = saturation_scale is not None
    if saturation_scale is not None and saturation_scale <= 0:
        raise ValueError("saturation_scale must be positive")

    contribs = compute_contribution_arrays(
        U_raw=U_tactical,
        delta=delta,
        beta=beta_leaf,
        beta_level=leaf_level,
        hierarchy=hierarchy,
        leaf_level=leaf_level,
        normalize_adstock=normalize_adstock,
        use_saturation=use_saturation,
        s_sat=None if saturation_scale is None else float(saturation_scale),
    )

    tactical_contrib = contribs[leaf_level]
    platform_contrib = (
        contribs.get(platform_level, np.zeros((T, len(platform_names))))
        if platform_level is not None
        else np.zeros((T, len(platform_names)))
    )
    channel_contrib = (
        contribs.get(channel_level, np.zeros((T, len(channel_names))))
        if channel_level is not None
        else np.zeros((T, len(channel_names)))
    )

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

    intercept_contrib = np.full(T, float(beta0) + float(beta0_offset))
    top_level = hierarchy.levels[-1]
    marketing_total = (
        contribs[top_level].sum(axis=1)
        if top_level in contribs
        else tactical_contrib.sum(axis=1)
    )
    fitted = intercept_contrib + marketing_total
    if control_contrib.size > 0:
        fitted = fitted + control_contrib.sum(axis=1)

    tactical_df = pd.DataFrame(tactical_contrib, index=time_index, columns=leaf_names)
    platform_df = pd.DataFrame(platform_contrib, index=time_index, columns=platform_names)
    channel_df = pd.DataFrame(channel_contrib, index=time_index, columns=channel_names)
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
    "beta_per_leaf_from_params",
    "beta_per_tactical_from_params",
]
