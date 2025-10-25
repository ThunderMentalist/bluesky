"""Precision MMM Bayesian MCMC package with lazy attribute loading."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Hierarchy": ("precision.precision.hierarchy", "Hierarchy"),
    "build_hierarchy": ("precision.precision.hierarchy", "build_hierarchy"),
    "Priors": ("precision.precision.priors", "Priors"),
    "MediaStandardize": ("precision.precision.priors", "MediaStandardize"),
    "BetaStructure": ("precision.precision.priors", "BetaStructure"),
    "SparsityPrior": ("precision.precision.priors", "SparsityPrior"),
    "LikelihoodFamily": ("precision.precision.priors", "LikelihoodFamily"),
    "ResidualMode": ("precision.precision.priors", "ResidualMode"),
    "adstock_geometric_np": ("precision.precision.adstock", "adstock_geometric_np"),
    "adstock_geometric_tf": ("precision.precision.adstock", "adstock_geometric_tf"),
    "ParamSpec": ("precision.precision.posterior", "ParamSpec"),
    "make_target_log_prob_fn": ("precision.precision.posterior", "make_target_log_prob_fn"),
    "PosteriorSamples": ("precision.precision.sampling", "PosteriorSamples"),
    "run_nuts": ("precision.precision.sampling", "run_nuts"),
    "Contributions": ("precision.precision.summaries", "Contributions"),
    "compute_contributions_from_params": (
        "precision.precision.summaries",
        "compute_contributions_from_params",
    ),
    "ROIResult": ("precision.precision.roi", "ROIResult"),
    "ROIInterval": ("precision.precision.roi", "ROIInterval"),
    "ROIIntervalTS": ("precision.precision.roi", "ROIIntervalTS"),
    "ROIUncertainty": ("precision.precision.roi", "ROIUncertainty"),
    "compute_roi": ("precision.precision.roi", "compute_roi"),
    "compute_roi_from_draws": ("precision.precision.roi", "compute_roi_from_draws"),
    "compute_roas": ("precision.precision.roi", "compute_roas"),
    "posterior_mean": ("precision.precision.summaries", "posterior_mean"),
    "summarise_decay_rates": ("precision.precision.summaries", "summarise_decay_rates"),
    "log_likelihood_draws_single_model": (
        "precision.precision.summaries",
        "log_likelihood_draws_single_model",
    ),
    "fourier_seasonality": ("precision.precision.controls", "fourier_seasonality"),
    "holiday_flags": ("precision.precision.controls", "holiday_flags"),
    "stack_controls": ("precision.precision.controls", "stack_controls"),
    "fit_media_scales_pre_adstock_tactical": (
        "precision.precision.scaling",
        "fit_media_scales_pre_adstock_tactical",
    ),
    "apply_pre_adstock_tactical": (
        "precision.precision.scaling",
        "apply_pre_adstock_tactical",
    ),
    "center_y": ("precision.precision.scaling", "center_y"),
    "ensemble": ("precision.precision.ensemble", "ensemble"),
    "PerModelResult": ("precision.precision.ensemble", "PerModelResult"),
    "EnsembleResult": ("precision.precision.ensemble", "EnsembleResult"),
    "psis_loo_pointwise": ("precision.precision.psis", "psis_loo_pointwise"),
    "stacking_weights_from_pointwise_logdens": (
        "precision.precision.psis",
        "stacking_weights_from_pointwise_logdens",
    ),
}


__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised via import
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - simple delegation
        raise AttributeError(f"module 'precision.precision' has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(set(globals().keys()) | set(__all__))
