"""Model ensembling utilities for multi-metric MMM fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .hierarchy import Hierarchy
from .posterior import make_target_log_prob_fn
from .priors import Priors
from .sampling import run_nuts
from .summaries import (
    Contributions,
    compute_contributions_from_params,
    posterior_mean,
    summarise_decay_rates,
)

MetricName = str


@dataclass
class PerModelResult:
    """Container for the outputs of a single metric-specific MMM fit."""

    metric: MetricName
    post_mean: Dict[str, np.ndarray]
    decay_df: pd.DataFrame
    contributions: Contributions
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


def _fill_missing_metrics_per_tactical(
    U_metrics: Dict[MetricName, Optional[np.ndarray]],
    availability: Optional[Dict[MetricName, np.ndarray]],
    fallback_order: List[MetricName],
) -> Dict[MetricName, np.ndarray]:
    """Ensure each metric has a complete [T, N_t] matrix.

    If a metric is missing entirely or unavailable for specific tacticals, the
    corresponding values are copied from the first available metric defined by
    ``fallback_order``.
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
            U_full[name] = ref_matrix.copy()
        else:
            if matrix.shape != (T, num_tacticals):
                raise ValueError(
                    f"U_metrics[{name}] has shape {matrix.shape}, expected {(T, num_tacticals)}"
                )
            U_full[name] = matrix.copy()

    for tactical_idx in range(num_tacticals):
        for name in metric_keys:
            metric_availability = availability.get(name)
            if metric_availability is not None and not bool(metric_availability[tactical_idx]):
                replaced = False
                for fallback in fallback_order:
                    fallback_avail = availability.get(fallback)
                    has_fallback = True
                    if fallback_avail is not None:
                        has_fallback = bool(fallback_avail[tactical_idx])
                    if has_fallback and U_full.get(fallback) is not None:
                        U_full[name][:, tactical_idx] = U_full[fallback][:, tactical_idx]
                        replaced = True
                        break
                if not replaced:
                    for fallback in fallback_order:
                        if U_full.get(fallback) is not None:
                            U_full[name][:, tactical_idx] = U_full[fallback][:, tactical_idx]
                            replaced = True
                            break
                if not replaced:
                    raise RuntimeError("Unable to fill missing tactical column for metric availability.")

    return U_full


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

    target_log_prob_fn, dims = make_target_log_prob_fn(
        y=y,
        U_tactical=U,
        Z_controls=Z,
        hierarchy=hierarchy,
        normalize_adstock=normalize_adstock,
        priors=priors,
    )
    samples = run_nuts(target_log_prob_fn, dims, **nuts_args)
    post_mean = posterior_mean(samples)
    decay_df = summarise_decay_rates(samples, hierarchy)

    contributions = compute_contributions_from_params(
        y=y,
        U_tactical=U,
        Z_controls=Z,
        control_names=control_names,
        hierarchy=hierarchy,
        beta0=post_mean["beta0"],
        beta_channel=post_mean["beta_channel"],
        gamma=post_mean["gamma"],
        delta=post_mean["delta"],
        normalize_adstock=normalize_adstock,
    )

    residuals = y - contributions.fitted.values
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(residuals**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    sigma_mean = float(post_mean["sigma"])

    return PerModelResult(
        metric=metric,
        post_mean=post_mean,
        decay_df=decay_df,
        contributions=contributions,
        r2=r2,
        rmse=rmse,
        mae=mae,
        sigma_mean=sigma_mean,
        weight=np.nan,
    )


def _compute_weights(
    per_model: Dict[MetricName, PerModelResult],
    scheme: str,
    power: float,
) -> Dict[MetricName, float]:
    """Compute model weights under the requested ``scheme``."""

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
            raise ValueError("Unknown weighting scheme. Use 'r2', 'rmse', 'mae', 'sigma', or 'uniform'.")

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


def ensemble(
    *,
    hierarchy: Hierarchy,
    y: np.ndarray,
    U_metrics: Dict[MetricName, Optional[np.ndarray]],
    Z_controls: Optional[np.ndarray] = None,
    control_names: Optional[List[str]] = None,
    availability: Optional[Dict[MetricName, np.ndarray]] = None,
    fallback_order: Optional[List[MetricName]] = None,
    normalize_adstock: bool = False,
    priors: Optional[Priors] = None,
    nuts_args: Optional[Dict] = None,
    weight_scheme: str = "r2",
    weight_power: float = 1.0,
) -> EnsembleResult:
    """Fit per-metric MMMs and aggregate their contributions.

    Parameters mirror the lower-level modelling utilities but allow for three
    media metric matrices (impressions, clicks, conversions). For tacticals
    where a metric is unavailable, provide ``availability`` so the data can be
    backfilled from other metrics following ``fallback_order``.
    """

    priors = priors or Priors()
    fallback_order = fallback_order or ["impressions", "clicks", "conversions"]
    nuts_defaults = dict(num_chains=3, num_burnin=1000, num_samples=1000, init_step_size=0.1, seed=123)
    nuts_args = nuts_args or {}
    full_nuts_args = {**nuts_defaults, **nuts_args}

    U_full = _fill_missing_metrics_per_tactical(U_metrics, availability, fallback_order)

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

    return EnsembleResult(
        per_model=per_model,
        weights=weights,
        contributions_weighted=contributions_weighted,
    )


__all__ = ["ensemble", "EnsembleResult", "PerModelResult"]
