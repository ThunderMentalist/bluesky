"""Prior configuration for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DecayMode = Literal["beta", "half_life", "hier_logit"]
LikelihoodFamily = Literal["normal", "student_t"]
ResidualMode = Literal["iid", "ar1"]
BetaStructure = Literal["channel", "platform_hier", "tactical_hier"]
SparsityPrior = Literal["none", "horseshoe", "dl"]


@dataclass(frozen=True)
class Priors:
    """Container for prior hyperparameters and decay configuration."""

    beta0_sd: float = 5.0
    beta_sd: float = 2.0
    gamma_sd: float = 2.0
    sigma_sd: float = 1.0

    # Legacy Beta(alpha, beta) for delta (used when decay_mode="beta").
    beta_alpha: float = 2.0
    beta_beta: float = 5.0

    # Decay modelling mode (defaults to half-life parameterisation).
    decay_mode: DecayMode = "half_life"

    # Half-life h ~ LogNormal(log_mu, log_sd); delta = 2**(-1 / h).
    half_life_log_mu: float = float(np.log(4.0))  # 4 weeks median half-life
    half_life_log_sd: float = 0.5

    # Hierarchical prior on logit(delta) ~ Normal(mu_c, tau_c^2).
    # Hyperpriors governing per-channel means and scales.
    hier_mu0: float = float(np.log(0.3 / 0.7))
    hier_mu0_sd: float = 1.0
    hier_tau_sd: float = 0.5  # HalfNormal scale for tau_c

    # Likelihood / residual configuration.
    likelihood: LikelihoodFamily = "normal"
    student_t_df: float = 5.0
    residual_mode: ResidualMode = "iid"
    phi_prior_sd: float = 0.5

    # Structure for beta coefficients and optional sparsity.
    beta_structure: BetaStructure = "platform_hier"
    sparsity_prior: SparsityPrior = "none"

    # Hierarchical pooling / sparsity hyperparameters.
    beta_pool_sd: float = 1.0
    hs_global_scale: float = 0.5
    hs_slab_scale: float = 1.0
    hs_slab_df: float = 4.0

    # --- Saturation configuration ---
    use_saturation: bool = False
    sat_log_mu: float = float(np.log(10.0))
    sat_log_sd: float = 0.7

    # --- Positive beta configuration ---
    enforce_beta_positive: bool = True
    beta_log_mu: float = float(np.log(0.1))
    beta_log_sd: float = 1.0


__all__ = [
    "BetaStructure",
    "DecayMode",
    "LikelihoodFamily",
    "ResidualMode",
    "Priors",
    "SparsityPrior",
]
