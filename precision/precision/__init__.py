"""Precision MMM Bayesian MCMC package."""

from .hierarchy import Hierarchy, build_hierarchy
from .priors import Priors
from .adstock import adstock_geometric_np, adstock_geometric_tf
from .posterior import ParamSpec, make_target_log_prob_fn
from .sampling import PosteriorSamples, run_nuts
from .summaries import (
    Contributions,
    compute_contributions_from_params,
    posterior_mean,
    summarise_decay_rates,
)
from .ensemble import EnsembleResult, PerModelResult, ensemble

__all__ = [
    "Hierarchy",
    "build_hierarchy",
    "Priors",
    "adstock_geometric_np",
    "adstock_geometric_tf",
    "ParamSpec",
    "make_target_log_prob_fn",
    "PosteriorSamples",
    "run_nuts",
    "Contributions",
    "compute_contributions_from_params",
    "posterior_mean",
    "summarise_decay_rates",
    "ensemble",
    "PerModelResult",
    "EnsembleResult",
]
