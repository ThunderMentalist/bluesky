"""Namespace package entry point for the Precision MMM toolkit."""

from importlib import import_module
from typing import Any

__all__ = [
    "Hierarchy",
    "build_hierarchy",
    "Priors",
    "BetaStructure",
    "SparsityPrior",
    "LikelihoodFamily",
    "ResidualMode",
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
    "fourier_seasonality",
    "holiday_flags",
    "stack_controls",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    if name in __all__:
        module = import_module(".precision", __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'precision' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(set(globals().keys()) | set(__all__))
