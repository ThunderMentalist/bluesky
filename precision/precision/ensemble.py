"""Model ensembling utilities for multi-metric MMM fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import warnings

from .hierarchy import Hierarchy
from .posterior import make_target_log_prob_fn
from .priors import Priors
from .sampling import PosteriorSamples, run_nuts
from .summaries import (
    ContributionInterval,
    ContributionIntervalSeries,
    ContributionUncertainty,
    Contributions,
    compute_contribution_arrays,
    _summarise_draws,
    _summarise_draws_series,
    posterior_mean,
    summarise_decay_rates,
)
from .psis import psis_loo_pointwise, stacking_weights_from_pointwise_logdens

MetricName = str


@dataclass
class PerModelResult:
    """Container for the outputs of a single metric-specific MMM fit."""

    metric: MetricName
    post_mean: Dict[str, np.ndarray]
    decay_df: pd.DataFrame
    contributions: Contributions
    uncertainty: ContributionUncertainty
    log_likelihood_draws: np.ndarray
    r2: float
    rmse: float
    mae: float
    sigma_mean: float
    weight: float


@dataclass
class EnsembleResult:
    """Result of combining metric-specific models via weighted averaging."""

    per_model: Dict[MetricName, PerModelResult]
    weights: Dict[MetricName, float]
    contributions_weighted: Contributions
    uncertainty_weighted: ContributionUncertainty


def _shift_with_lag(U: np.ndarray, lag: int) -> np.ndarray:
    """Shift matrix ``U`` forward by ``lag`` periods, padding with zeros."""

    if lag <= 0:
        return U
    T, N = U.shape
    out = np.zeros_like(U)
    if lag < T:
        out[lag:, :] = U[:-lag, :]
    return out


def _apply_metric_lags(
    U_metrics: Dict[MetricName, Optional[np.ndarray]],
    metric_lags: Optional[Dict[MetricName, int]],
) -> Dict[MetricName, Optional[np.ndarray]]:
    if not metric_lags:
        return U_metrics
    out: Dict[MetricName, Optional[np.ndarray]] = {}
    for name, matrix in U_metrics.items():
        if matrix is None:
            out[name] = None
            continue
        lag = int(metric_lags.get(name, 0))
        out[name] = _shift_with_lag(matrix, lag)
    return out


def _check_metric_overlap(
    y: np.ndarray,
    U: Optional[np.ndarray],
    name: str,
    *,
    corr_thresh: float = 0.97,
) -> None:
    """Basic leakage guardrail based on correlation with the outcome."""

    if U is None:
        return
    if U.shape[0] != y.shape[0]:
        return
    aggregate = U.sum(axis=1)
    y_std = (y - y.mean()) / (y.std(ddof=0) + 1e-9)
    agg_std = (aggregate - aggregate.mean()) / (aggregate.std(ddof=0) + 1e-9)
    corr = float(np.clip(np.corrcoef(y_std, agg_std)[0, 1], -1.0, 1.0))
    if np.isfinite(corr) and abs(corr) >= corr_thresh:
        raise ValueError(
            "Potential leakage detected: metric "
            f"'{name}' correlates {corr:.3f} with the dependent variable. "
            "Ensure the metric is pre-treatment or lagged appropriately."
        )


def _fill_missing_metrics_per_tactical(
    U_metrics: Dict[MetricName, Optional[np.ndarray]],
    availability: Optional[Dict[MetricName, np.ndarray]],
    fallback_order: List[MetricName],
) -> Dict[MetricName, np.ndarray]:
    """Ensure each metric has a complete [T, N_t] matrix.

    If a metric is missing entirely, a zero matrix of matching shape is used. For
    tacticals where a metric is unavailable according to ``availability`` the
    corresponding columns are set to zero rather than copied from other metrics,
    avoiding duplicated exposure signals across models.
    """

    metric_keys = ["impressions", "clicks", "conversions"]
    availability = availability or {}

    ref_matrix: Optional[np.ndarray] = None
    for name in fallback_order:
        candidate = U_metrics.get(name)
        if candidate is not None:
            ref_matrix = candidate
            break
    if ref_matrix is None:
        raise ValueError("At least one metric matrix must be provided in U_metrics.")

    T, num_tacticals = ref_matrix.shape

    U_full: Dict[MetricName, np.ndarray] = {}
    for name in metric_keys:
        matrix = U_metrics.get(name)
        if matrix is None:
            matrix = np.zeros_like(ref_matrix)
        else:
            if matrix.shape != (T, num_tacticals):
                raise ValueError(
                    f"U_metrics[{name}] has shape {matrix.shape}, expected {(T, num_tacticals)}"
                )
            matrix = matrix.copy()

        metric_availability = availability.get(name)
        if metric_availability is not None:
            if metric_availability.shape[0] != num_tacticals:
                raise ValueError(
                    f"availability[{name}] has length {metric_availability.shape[0]}, expected {num_tacticals}"
                )
            mask = metric_availability.astype(bool)
            if not np.all(mask):
                matrix[:, ~mask] = 0.0
        U_full[name] = matrix

    return U_full


def _get_control_names(control_names: Optional[List[str]], num_controls: int) -> List[str]:
    if num_controls == 0:
        return []
    if control_names is None:
        return [f"control_{idx}" for idx in range(num_controls)]
    if len(control_names) != num_controls:
        raise ValueError(
            f"Expected {num_controls} control names, received {len(control_names)}."
        )
    return list(control_names)


def _arrays_to_dataframe(
    values: np.ndarray,
    index: pd.Index,
    columns: List[str],
) -> pd.DataFrame:
    if values.size == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    return pd.DataFrame(values, index=index, columns=columns)


def _arrays_to_series(values: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    return pd.Series(values, index=index, name=name)


def _make_contributions_from_arrays(
    *,
    hierarchy: Hierarchy,
    control_names: List[str],
    tactical: np.ndarray,
    platform: np.ndarray,
    channel: np.ndarray,
    controls: np.ndarray,
    intercept: np.ndarray,
    fitted: np.ndarray,
) -> Contributions:
    index = pd.RangeIndex(tactical.shape[0], name="time")
    tactical_df = _arrays_to_dataframe(tactical, index, hierarchy.tactical_names)
    platform_df = _arrays_to_dataframe(platform, index, hierarchy.platform_names)
    channel_df = _arrays_to_dataframe(channel, index, hierarchy.channel_names)
    controls_df = _arrays_to_dataframe(controls, index, control_names)
    intercept_series = _arrays_to_series(intercept, index, "intercept")
    fitted_series = _arrays_to_series(fitted, index, "fitted")

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


def _fit_single_metric_model(
    metric: MetricName,
    y: np.ndarray,
    U: np.ndarray,
    Z: Optional[np.ndarray],
    control_names: Optional[List[str]],
    hierarchy: Hierarchy,
    *,
    normalize_adstock: bool,
    priors: Priors,
    nuts_args: Dict,
) -> PerModelResult:
    """Fit a single metric-specific MMM and compute summary statistics."""

    target_log_prob_fn, dims, param_spec = make_target_log_prob_fn(
        y=y,
        U_tactical=U,
        Z_controls=Z,
        hierarchy=hierarchy,
        normalize_adstock=normalize_adstock,
        priors=priors,
    )
    samples = run_nuts(target_log_prob_fn, dims, param_spec, **nuts_args)
    post_mean = posterior_mean(samples)
    decay_df = summarise_decay_rates(samples, hierarchy)

    stacked: PosteriorSamples = samples.stack_chains()
    num_draws = stacked.beta0.shape[0]
    T, num_tacticals = U.shape
    num_platforms = hierarchy.num_platforms
    num_channels = hierarchy.num_channels
    num_controls = 0 if Z is None else Z.shape[1]
    control_names_full = _get_control_names(control_names, num_controls)

    tactical_draws = np.zeros((num_draws, T, num_tacticals))
    platform_draws = np.zeros((num_draws, T, num_platforms))
    channel_draws = np.zeros((num_draws, T, num_channels))
    controls_draws = np.zeros((num_draws, T, num_controls))
    intercept_draws = np.zeros((num_draws, T))
    fitted_draws = np.zeros((num_draws, T))

    log_likelihood_draws = np.zeros((num_draws, T))

    for idx in range(num_draws):
        beta_channel = stacked.beta_channel[idx] if stacked.beta_channel.size > 0 else None
        beta_platform = None if stacked.beta_platform is None else stacked.beta_platform[idx]
        beta_tactical = None if stacked.beta_tactical is None else stacked.beta_tactical[idx]
        gamma_draw = None
        if stacked.gamma is not None and stacked.gamma.size > 0:
            gamma_draw = stacked.gamma[idx]
        tactical, platform, channel, controls_arr, intercept, fitted = compute_contribution_arrays(
            U,
            Z,
            hierarchy,
            beta0=float(stacked.beta0[idx]),
            beta_channel=beta_channel,
            gamma=gamma_draw,
            delta=stacked.delta[idx],
            normalize_adstock=normalize_adstock,
            beta_platform=beta_platform,
            beta_tactical=beta_tactical,
        )

        tactical_draws[idx] = tactical
        platform_draws[idx] = platform
        channel_draws[idx] = channel
        if num_controls > 0:
            controls_draws[idx] = controls_arr
        intercept_draws[idx] = intercept
        fitted_draws[idx] = fitted

        sigma = max(float(stacked.sigma[idx]), 1e-9)
        resid = y - fitted
        log_likelihood_draws[idx] = -0.5 * (
            (resid**2) / (sigma**2) + np.log(2.0 * np.pi * (sigma**2))
        )

    index = pd.RangeIndex(T, name="time")
    contributions_mean = _make_contributions_from_arrays(
        hierarchy=hierarchy,
        control_names=control_names_full,
        tactical=tactical_draws.mean(axis=0),
        platform=platform_draws.mean(axis=0),
        channel=channel_draws.mean(axis=0),
        controls=controls_draws.mean(axis=0) if num_controls > 0 else controls_draws.mean(axis=0),
        intercept=intercept_draws.mean(axis=0),
        fitted=fitted_draws.mean(axis=0),
    )

    uncertainty = ContributionUncertainty(
        tactical=_summarise_draws(tactical_draws, index, hierarchy.tactical_names),
        platform=_summarise_draws(platform_draws, index, hierarchy.platform_names),
        channel=_summarise_draws(channel_draws, index, hierarchy.channel_names),
        controls=_summarise_draws(controls_draws, index, control_names_full),
        intercept=_summarise_draws_series(intercept_draws, index, "intercept"),
        fitted=_summarise_draws_series(fitted_draws, index, "fitted"),
    )

    residuals_mean = y - contributions_mean.fitted.values
    rmse = float(np.sqrt(np.mean(residuals_mean**2)))
    mae = float(np.mean(np.abs(residuals_mean)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(residuals_mean**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    sigma_mean = float(post_mean["sigma"])

    return PerModelResult(
        metric=metric,
        post_mean=post_mean,
        decay_df=decay_df,
        contributions=contributions_mean,
        uncertainty=uncertainty,
        log_likelihood_draws=log_likelihood_draws,
        r2=r2,
        rmse=rmse,
        mae=mae,
        sigma_mean=sigma_mean,
        weight=np.nan,
    )


def _stacking_weights_from_loglik(
    loglik: Dict[MetricName, np.ndarray],
    *,
    max_iter: int = 2000,
    lr: float = 0.1,
    tol: float = 1e-7,
) -> Dict[MetricName, float]:
    """Exponentiated-gradient solver for stacking weights."""

    keys = list(loglik.keys())
    if not keys:
        return {}

    log_p = []
    for key in keys:
        draws = loglik[key]
        if draws.ndim != 2:
            raise ValueError("log_likelihood_draws must have shape [draws, T].")
        # log predictive density per observation via log-mean-exp across draws
        max_draw = np.max(draws, axis=0, keepdims=True)
        log_mean = max_draw + np.log(np.mean(np.exp(draws - max_draw), axis=0, keepdims=True))
        log_p.append(log_mean.squeeze(0))
    log_p_matrix = np.stack(log_p, axis=0)  # [M, T]

    w = np.ones(len(keys)) / len(keys)
    previous = -np.inf
    for _ in range(max_iter):
        max_log = np.max(log_p_matrix, axis=0, keepdims=True)
        scaled = np.exp(log_p_matrix - max_log)
        denom = np.dot(w, scaled)
        denom = np.clip(denom, 1e-300, np.inf)
        objective = float(np.sum(np.log(denom) + max_log.squeeze(0)))
        grad = np.sum(scaled / denom, axis=1)
        w_new = w * np.exp(lr * grad)
        w_new = np.clip(w_new, 1e-12, np.inf)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            previous = objective
            break
        w, previous = w_new, objective

    if not np.all(np.isfinite(w)):
        w = np.ones(len(keys)) / len(keys)

    return {keys[idx]: float(w[idx]) for idx in range(len(keys))}


def _compute_weights_psis_loo(
    per_model: Dict[MetricName, PerModelResult]
) -> Dict[MetricName, float]:
    """Compute stacking weights by maximising PSIS-LOO predictive density."""

    if not per_model:
        return {}

    pointwise: Dict[MetricName, np.ndarray] = {}
    for name, result in per_model.items():
        loo_lpd, pareto_k = psis_loo_pointwise(result.log_likelihood_draws)
        pointwise[name] = loo_lpd
        if np.mean(pareto_k > 0.7) > 0.1:
            warnings.warn(
                (
                    "PSIS-LOO may be unstable for metric "
                    f"'{name}' (k > 0.7 in more than 10% of time points)."
                ),
                RuntimeWarning,
            )

    weights = stacking_weights_from_pointwise_logdens(pointwise)
    if not weights:
        uniform = 1.0 / len(per_model)
        return {name: uniform for name in per_model}
    return weights


def _compute_weights(
    per_model: Dict[MetricName, PerModelResult],
    scheme: str,
    power: float,
) -> Dict[MetricName, float]:
    """Compute model weights under the requested ``scheme``."""

    if scheme == "psis_loo":
        return _compute_weights_psis_loo(per_model)
    if scheme == "stacking":
        weights = _stacking_weights_from_loglik(
            {name: result.log_likelihood_draws for name, result in per_model.items()}
        )
        if weights:
            return weights

    metrics: Dict[MetricName, float] = {}
    for name, result in per_model.items():
        if scheme == "r2":
            metrics[name] = max(0.0, result.r2)
        elif scheme == "rmse":
            metrics[name] = 1.0 / max(1e-9, result.rmse)
        elif scheme == "mae":
            metrics[name] = 1.0 / max(1e-9, result.mae)
        elif scheme == "sigma":
            metrics[name] = 1.0 / max(1e-9, result.sigma_mean)
        elif scheme == "uniform":
            metrics[name] = 1.0
        else:
            raise ValueError(
                "Unknown weighting scheme. Use 'psis_loo', 'stacking', 'r2', 'rmse', 'mae', 'sigma', or 'uniform'."
            )

    raw = {name: value**power for name, value in metrics.items()}
    total = sum(raw.values())
    if total <= 0 or not np.isfinite(total):
        uniform_weight = 1.0 / len(per_model)
        return {name: uniform_weight for name in per_model}

    return {name: value / total for name, value in raw.items()}


def _weighted_sum_frames(
    frames: Dict[MetricName, pd.DataFrame],
    weights: Dict[MetricName, float],
) -> pd.DataFrame:
    result: Optional[pd.DataFrame] = None
    for name, frame in frames.items():
        weighted = frame * weights[name]
        result = weighted if result is None else result + weighted
    if result is None:
        raise ValueError("No frames provided for aggregation.")
    return result


def _weighted_sum_series(
    series: Dict[MetricName, pd.Series],
    weights: Dict[MetricName, float],
) -> pd.Series:
    result: Optional[pd.Series] = None
    for name, values in series.items():
        weighted = values * weights[name]
        result = weighted if result is None else result + weighted
    if result is None:
        raise ValueError("No series provided for aggregation.")
    return result


def _aggregate_contributions(
    per_model: Dict[MetricName, PerModelResult],
    weights: Dict[MetricName, float],
) -> Contributions:
    """Weighted average of contributions across models."""

    tactical = _weighted_sum_frames({k: v.contributions.tactical for k, v in per_model.items()}, weights)
    platform = _weighted_sum_frames({k: v.contributions.platform for k, v in per_model.items()}, weights)
    channel = _weighted_sum_frames({k: v.contributions.channel for k, v in per_model.items()}, weights)
    controls = _weighted_sum_frames({k: v.contributions.controls for k, v in per_model.items()}, weights)
    intercept = _weighted_sum_series({k: v.contributions.intercept for k, v in per_model.items()}, weights)
    fitted = _weighted_sum_series({k: v.contributions.fitted for k, v in per_model.items()}, weights)

    return Contributions(
        tactical=tactical,
        platform=platform,
        channel=channel,
        controls=controls,
        intercept=intercept,
        fitted=fitted,
        tactical_totals=tactical.sum(axis=0),
        platform_totals=platform.sum(axis=0),
        channel_totals=channel.sum(axis=0),
        controls_totals=controls.sum(axis=0),
        intercept_total=float(intercept.sum()),
        fitted_total=float(fitted.sum()),
    )


def _weighted_interval_frames(
    intervals: Dict[MetricName, ContributionInterval],
    weights: Dict[MetricName, float],
) -> ContributionInterval:
    lower: Optional[pd.DataFrame] = None
    upper: Optional[pd.DataFrame] = None
    for name, interval in intervals.items():
        w = weights[name]
        if lower is None:
            lower = interval.lower * w
            upper = interval.upper * w
        else:
            lower = lower + interval.lower * w
            upper = upper + interval.upper * w
    if lower is None or upper is None:
        raise ValueError("No intervals provided for aggregation.")
    return ContributionInterval(lower=lower, upper=upper)


def _weighted_interval_series(
    intervals: Dict[MetricName, ContributionIntervalSeries],
    weights: Dict[MetricName, float],
) -> ContributionIntervalSeries:
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None
    for name, interval in intervals.items():
        w = weights[name]
        if lower is None:
            lower = interval.lower * w
            upper = interval.upper * w
        else:
            lower = lower + interval.lower * w
            upper = upper + interval.upper * w
    if lower is None or upper is None:
        raise ValueError("No intervals provided for aggregation.")
    return ContributionIntervalSeries(lower=lower, upper=upper)


def _aggregate_uncertainty(
    per_model: Dict[MetricName, PerModelResult],
    weights: Dict[MetricName, float],
) -> ContributionUncertainty:
    return ContributionUncertainty(
        tactical=_weighted_interval_frames({k: v.uncertainty.tactical for k, v in per_model.items()}, weights),
        platform=_weighted_interval_frames({k: v.uncertainty.platform for k, v in per_model.items()}, weights),
        channel=_weighted_interval_frames({k: v.uncertainty.channel for k, v in per_model.items()}, weights),
        controls=_weighted_interval_frames({k: v.uncertainty.controls for k, v in per_model.items()}, weights),
        intercept=_weighted_interval_series({k: v.uncertainty.intercept for k, v in per_model.items()}, weights),
        fitted=_weighted_interval_series({k: v.uncertainty.fitted for k, v in per_model.items()}, weights),
    )


def ensemble(
    *,
    hierarchy: Hierarchy,
    y: np.ndarray,
    U_metrics: Dict[MetricName, Optional[np.ndarray]],
    Z_controls: Optional[np.ndarray] = None,
    control_names: Optional[List[str]] = None,
    availability: Optional[Dict[MetricName, np.ndarray]] = None,
    fallback_order: Optional[List[MetricName]] = None,
    normalize_adstock: bool = True,
    priors: Optional[Priors] = None,
    nuts_args: Optional[Dict] = None,
    weight_scheme: str = "psis_loo",
    weight_power: float = 1.0,
    metric_lags: Optional[Dict[MetricName, int]] = None,
    offline_channels: Optional[List[str]] = None,
    enforce_offline_exclusion: bool = True,
    leakage_corr_threshold: float = 0.97,
) -> EnsembleResult:
    """Fit per-metric MMMs and aggregate their contributions.

    Parameters mirror the lower-level modelling utilities but allow for three
    media metric matrices (impressions, clicks, conversions). For tacticals
    where a metric is unavailable, provide ``availability`` to zero out the
    corresponding columns instead of copying exposure from another metric.
    Additional safeguards include per-metric lags, leakage checks, offline
    channel exclusion for secondary metrics, and PSIS-LOO-based stacking
    weights.
    """

    priors = priors or Priors()
    fallback_order = fallback_order or ["impressions", "clicks", "conversions"]
    nuts_defaults = dict(num_chains=3, num_burnin=1000, num_samples=1000, init_step_size=0.1, seed=123)
    nuts_args = nuts_args or {}
    full_nuts_args = {**nuts_defaults, **nuts_args}

    # Step 1: metric pre-processing
    U_lagged = _apply_metric_lags(U_metrics, metric_lags)
    for metric_name in ["impressions", "clicks", "conversions"]:
        _check_metric_overlap(
            y,
            U_lagged.get(metric_name),
            metric_name,
            corr_thresh=leakage_corr_threshold,
        )

    availability_masks = availability.copy() if availability is not None else {}
    if enforce_offline_exclusion and offline_channels:
        offline_set = set(offline_channels)
        offline_tacticals = np.array(
            [
                hierarchy.channel_names[hierarchy.p_to_c[hierarchy.t_to_p[t]]] in offline_set
                for t in range(hierarchy.num_tacticals)
            ],
            dtype=bool,
        )
        for secondary_metric in ["clicks", "conversions"]:
            existing = availability_masks.get(secondary_metric)
            if existing is None:
                mask = ~offline_tacticals
            else:
                if existing.shape[0] != hierarchy.num_tacticals:
                    raise ValueError(
                        "availability mask for metric '"
                        + secondary_metric
                        + "' has incorrect length."
                    )
                mask = existing.astype(bool) & (~offline_tacticals)
            availability_masks[secondary_metric] = mask.astype(np.uint8)

    U_full = _fill_missing_metrics_per_tactical(U_lagged, availability_masks, fallback_order)

    per_model: Dict[MetricName, PerModelResult] = {}
    for metric_name in ["impressions", "clicks", "conversions"]:
        per_model[metric_name] = _fit_single_metric_model(
            metric=metric_name,
            y=y,
            U=U_full[metric_name],
            Z=Z_controls,
            control_names=control_names,
            hierarchy=hierarchy,
            normalize_adstock=normalize_adstock,
            priors=priors,
            nuts_args=full_nuts_args,
        )

    weights = _compute_weights(per_model, scheme=weight_scheme, power=weight_power)
    for metric_name, result in per_model.items():
        result.weight = float(weights[metric_name])

    contributions_weighted = _aggregate_contributions(per_model, weights)
    uncertainty_weighted = _aggregate_uncertainty(per_model, weights)

    return EnsembleResult(
        per_model=per_model,
        weights=weights,
        contributions_weighted=contributions_weighted,
        uncertainty_weighted=uncertainty_weighted,
    )


__all__ = ["ensemble", "EnsembleResult", "PerModelResult"]
