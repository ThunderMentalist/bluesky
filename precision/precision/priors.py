"""Prior configuration for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Priors:
    """Container for weakly-informative prior hyperparameters."""

    beta0_sd: float = 5.0
    beta_sd: float = 2.0
    gamma_sd: float = 2.0
    sigma_sd: float = 1.0
    beta_alpha: float = 2.0
    beta_beta: float = 5.0


__all__ = ["Priors"]
