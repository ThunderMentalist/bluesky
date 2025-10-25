"""Prior configuration for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

DecayMode = Literal["beta", "half_life", "hier_logit"]
LikelihoodFamily = Literal["normal", "student_t"]
ResidualMode = Literal["iid", "ar1"]
BetaStructure = Literal["channel", "platform_hier", "tactical_hier"]
SparsityPrior = Literal["none", "horseshoe"]
MediaStandardize = Literal["none", "pre_adstock_tactical"]


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

    # ---- I: β structure & shrinkage ----
    beta_structure: BetaStructure = "platform_hier"
    leaf_level: Optional[str] = None
    beta_level: Optional[str] = None
    pool_parent_level: Optional[str] = None
    sparsity_prior: SparsityPrior = "none"
    beta_pool_sd: float = 0.75
    hs_global_scale: float = 0.5
    hs_slab_scale: float = 1.0
    hs_slab_df: float = 4.0

    # ---- J: interpretable lift priors (log-space) ----
    lift_log_mu: float = float(np.log(0.1))
    lift_log_sd: float = 1.0

    # ---- J: scaling options ----
    standardize_media: MediaStandardize = "pre_adstock_tactical"
    media_scale_stat: Literal["median", "mean", "p95"] = "median"
    center_y: bool = True

    # --- Saturation configuration ---
    use_saturation: bool = False
    sat_log_mu: float = float(np.log(10.0))
    sat_log_sd: float = 0.7

    def resolve_beta_levels(self, hierarchy_levels: List[str]) -> tuple[str, str, Optional[str]]:
        """Resolve the β pooling levels based on configuration and hierarchy."""

        leaf = self.leaf_level or hierarchy_levels[0]

        if self.beta_level:
            return leaf, self.beta_level, self.pool_parent_level

        if self.beta_structure == "channel":
            return leaf, hierarchy_levels[2], None

        if self.beta_structure == "platform_hier":
            return leaf, hierarchy_levels[1], hierarchy_levels[2]

        if self.beta_structure == "tactical_hier":
            return leaf, leaf, hierarchy_levels[2]

        raise ValueError(f"Unknown beta_structure={self.beta_structure!r}")


__all__ = [
    "BetaStructure",
    "DecayMode",
    "LikelihoodFamily",
    "MediaStandardize",
    "ResidualMode",
    "Priors",
    "SparsityPrior",
]
