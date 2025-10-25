"""Target log posterior construction for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .adstock import DTYPE, adstock_geometric_tf
from .constants import SAT_MIN, HALF_LIFE_MIN, SIGMA_MIN
from .hierarchy import Hierarchy
from .priors import Priors, normalize_beta_structure
from .scaling import (
    apply_pre_adstock_tactical,
    center_y as _center_y,
    fit_media_scales_pre_adstock_tactical,
)


tfd = tfp.distributions
tfb = tfp.bijectors


TargetLogProbFn = Callable[..., tf.Tensor]


@dataclass
class ParamSpec:
    """Specification for a model parameter used to construct NUTS kernels."""

    name: str
    shape: Tuple[int, ...]
    bijector: tfb.Bijector
    init: tf.Tensor


def _saturate_log1p(x: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
    """Apply the concave saturation transform elementwise."""

    s_safe = tf.maximum(s, tf.cast(SAT_MIN, DTYPE))
    return s_safe * tf.math.log1p(x / s_safe)


def make_target_log_prob_fn(
    y: np.ndarray,
    U_tactical: np.ndarray,
    Z_controls: Optional[np.ndarray],
    hierarchy: Hierarchy,
    *,
    normalize_adstock: bool = True,
    priors: Priors = Priors(),
) -> Tuple[TargetLogProbFn, Dict[str, int], List[ParamSpec]]:
    """
    Create a TensorFlow Probability target log-probability function.

    Returns:
      - target_log_prob: closure over named constrained parameters in `param_spec`.
      - dims: dict with keys (as available)
          T: int, time length
          N_t: int, # of leaf tacticals
          J: int, # of controls
          mode: str, decay mode {"beta","half_life","hier_logit"}
          residual_mode: str, {"iid","ar1"}
          beta_structure: str
          levels: list[str]
          level_sizes: dict[level->int]
          leaf_level: str
          beta_level: str
          pool_parent_level: Optional[str]
          tactical_rescale: np.ndarray (shape [N_t])
          y_mean: float (if priors.center_y)
          C, P: Optional[int] legacy counts if present
      - param_spec: list[ParamSpec] describing parameter bijectors and init values.
    """

    const = lambda value: tf.constant(value, dtype=DTYPE)

    T_, num_tacticals = U_tactical.shape
    if Z_controls is None:
        Z_tf = tf.zeros((T_, 0), dtype=DTYPE)
        num_controls = 0
    else:
        Z_tf = tf.convert_to_tensor(Z_controls, dtype=DTYPE)
        num_controls = int(Z_controls.shape[1])

    # Optional centring of the outcome for intercept interpretability.
    if priors.center_y:
        y_centered, y_mean = _center_y(y)
    else:
        y_centered, y_mean = y, 0.0
    y_tf = tf.convert_to_tensor(y_centered, dtype=DTYPE)

    # Optional standardisation of tactical inputs prior to adstocking.
    scale_mode = priors.standardize_media
    if scale_mode == "pre_adstock_tactical":
        scale_vec = fit_media_scales_pre_adstock_tactical(
            U_tactical, stat=priors.media_scale_stat
        )
        U_scaled = apply_pre_adstock_tactical(U_tactical, scale_vec)
        tactical_rescale = 1.0 / scale_vec
    elif scale_mode == "none":
        U_scaled = U_tactical
        tactical_rescale = np.ones(num_tacticals, dtype=float)
    else:
        raise ValueError(
            "priors.standardize_media must be 'none' or 'pre_adstock_tactical'"
        )

    U_tf = tf.convert_to_tensor(U_scaled, dtype=DTYPE)

    leaf_level, beta_level, pool_parent = priors.resolve_beta_levels(hierarchy.levels)
    # Early validation: pool parent must be a strict ancestor of beta_level if provided.
    if pool_parent is not None:
        child_idx = hierarchy.levels.index(beta_level)
        parent_idx = hierarchy.levels.index(pool_parent)
        if parent_idx <= child_idx:
            raise ValueError(
                f"pool_parent_level={pool_parent!r} must be an ancestor of beta_level={beta_level!r}"
            )
    level_sizes = {level: int(hierarchy.size(level)) for level in hierarchy.levels}
    M_leaf_to_beta = tf.constant(hierarchy.map(leaf_level, beta_level), dtype=DTYPE)
    child_to_parent: Optional[tf.Tensor] = None
    if pool_parent is not None:
        child_to_parent = tf.constant(
            hierarchy.index_map(beta_level, pool_parent), dtype=tf.int32
        )

    def _get_optional_size(attr: str, level_name: str) -> Optional[int]:
        try:
            return int(getattr(hierarchy, attr))
        except AttributeError:
            return level_sizes.get(level_name)

    C = _get_optional_size("num_channels", "channel")
    P = _get_optional_size("num_platforms", "platform")

    prior_beta0 = tfd.Normal(loc=const(0.0), scale=const(priors.beta0_sd))
    prior_gamma = (
        tfd.Normal(loc=const(0.0), scale=const(priors.gamma_sd))
        if num_controls > 0
        else None
    )
    prior_sigma = tfd.HalfNormal(scale=const(priors.sigma_sd))
    prior_s_sat = (
        tfd.LogNormal(loc=const(priors.sat_log_mu), scale=const(priors.sat_log_sd))
        if priors.use_saturation
        else None
    )
    prior_eta = tfd.Normal(loc=const(priors.lift_log_mu), scale=const(priors.lift_log_sd))
    prior_tau = tfd.HalfNormal(scale=const(priors.beta_pool_sd))
    prior_phi = tfd.Normal(loc=const(0.0), scale=const(priors.phi_prior_sd))

    beta_structure = normalize_beta_structure(priors.beta_structure)
    sparsity = priors.sparsity_prior
    if sparsity not in {"none", "horseshoe"}:
        raise ValueError(f"Unknown priors.sparsity_prior={sparsity!r}")
    if beta_structure != "tactical_hier" and sparsity != "none":
        raise ValueError("Sparsity priors are only supported with tactical_hier betas")
    if sparsity == "horseshoe" and beta_level != leaf_level:
        raise ValueError("Horseshoe sparsity requires beta_level to equal leaf_level")

    param_spec: List[ParamSpec] = []
    param_spec.append(ParamSpec("beta0", (), tfb.Identity(), tf.zeros([], DTYPE)))
    param_spec.append(
        ParamSpec("gamma", (num_controls,), tfb.Identity(), tf.zeros([num_controls], DTYPE))
    )
    if priors.use_saturation:
        param_spec.append(
            ParamSpec("s_sat", (), tfb.Exp(), const(priors.sat_log_mu))
        )
    if priors.residual_mode == "ar1":
        param_spec.append(ParamSpec("phi", (), tfb.Tanh(), tf.zeros([], DTYPE)))
    eta_parent_name: Optional[str] = None
    eta_beta_name = f"eta_{beta_level}"
    beta_size = level_sizes[beta_level]

    if pool_parent is None:
        param_spec.append(
            ParamSpec(
                eta_beta_name,
                (beta_size,),
                tfb.Identity(),
                tf.fill([beta_size], const(priors.lift_log_mu)),
            )
        )
    else:
        eta_parent_name = f"eta_{pool_parent}"
        parent_size = level_sizes[pool_parent]
        param_spec.append(
            ParamSpec(
                eta_parent_name,
                (parent_size,),
                tfb.Identity(),
                tf.fill([parent_size], const(priors.lift_log_mu)),
            )
        )
        if sparsity != "horseshoe":
            param_spec.append(
                ParamSpec(
                    "tau_beta",
                    (),
                    tfb.Softplus(),
                    const(priors.beta_pool_sd),
                )
            )
        param_spec.append(
            ParamSpec(
                eta_beta_name,
                (beta_size,),
                tfb.Identity(),
                tf.fill([beta_size], const(priors.lift_log_mu)),
            )
        )
        if sparsity == "horseshoe":
            param_spec.append(
                ParamSpec(
                    "tau0",
                    (),
                    tfb.Softplus(),
                    const(priors.hs_global_scale),
                )
            )
            param_spec.append(
                ParamSpec(
                    "lambda_local",
                    (beta_size,),
                    tfb.Softplus(),
                    tf.ones([beta_size], dtype=DTYPE),
                )
            )

    decay_specs: List[ParamSpec] = []
    decay_mode = priors.decay_mode

    if decay_mode == "beta":
        prior_delta = tfd.Beta(
            concentration1=const(priors.beta_alpha),
            concentration0=const(priors.beta_beta),
        )
        decay_specs.append(
            ParamSpec(
                "delta",
                (num_tacticals,),
                tfb.Sigmoid(),
                tf.fill([num_tacticals], const(0.30)),
            )
        )

        def delta_from_args(delta: tf.Tensor) -> tf.Tensor:
            return delta

        def log_prior_decay(delta: tf.Tensor) -> tf.Tensor:
            return tf.reduce_sum(prior_delta.log_prob(delta))

    elif decay_mode == "half_life":
        prior_log_h = tfd.Normal(
            loc=const(priors.half_life_log_mu),
            scale=const(priors.half_life_log_sd),
        )
        decay_specs.append(
            ParamSpec(
                "log_h",
                (num_tacticals,),
                tfb.Identity(),
                tf.fill([num_tacticals], const(priors.half_life_log_mu)),
            )
        )

        def delta_from_args(log_h: tf.Tensor) -> tf.Tensor:
            h = tf.exp(log_h)
            return tf.pow(const(2.0), -1.0 / tf.maximum(h, const(HALF_LIFE_MIN)))

        def log_prior_decay(log_h: tf.Tensor) -> tf.Tensor:
            return tf.reduce_sum(prior_log_h.log_prob(log_h))

    elif decay_mode == "hier_logit":
        prior_mu_c = tfd.Normal(loc=const(priors.hier_mu0), scale=const(priors.hier_mu0_sd))
        prior_tau_c = tfd.HalfNormal(scale=const(priors.hier_tau_sd))
        if C is None:
            raise ValueError("hier_logit decay requires a 'channel' level in the hierarchy")
        idx_leaf_to_channel_decay = tf.constant(
            hierarchy.index_map(leaf_level, "channel"), dtype=tf.int32
        )

        decay_specs.append(
            ParamSpec("logit_delta", (num_tacticals,), tfb.Identity(), tf.zeros([num_tacticals], DTYPE))
        )
        decay_specs.append(
            ParamSpec("mu_c", (C,), tfb.Identity(), tf.fill([C], const(priors.hier_mu0)))
        )
        decay_specs.append(
            ParamSpec(
                "log_tau_c",
                (C,),
                tfb.Identity(),
                tf.fill([C], tfp.math.softplus_inverse(const(0.5))),
            )
        )

        def delta_from_args(logit_delta: tf.Tensor, mu_c: tf.Tensor, log_tau_c: tf.Tensor) -> tf.Tensor:
            del mu_c
            del log_tau_c
            return tf.math.sigmoid(logit_delta)

        def log_prior_decay(logit_delta: tf.Tensor, mu_c: tf.Tensor, log_tau_c: tf.Tensor) -> tf.Tensor:
            tau_c = tf.math.softplus(log_tau_c)
            mu_i = tf.gather(mu_c, idx_leaf_to_channel_decay)
            tau_i = tf.gather(tau_c, idx_leaf_to_channel_decay)
            log_p = tf.reduce_sum(prior_mu_c.log_prob(mu_c))
            log_p += tf.reduce_sum(prior_tau_c.log_prob(tau_c))
            log_p += tf.reduce_sum(tf.math.log_sigmoid(log_tau_c))
            log_p += tf.reduce_sum(tfd.Normal(mu_i, tau_i).log_prob(logit_delta))
            return log_p

    else:
        raise ValueError(f"Unknown priors.decay_mode={decay_mode!r}")

    def _log_prior_common(
        beta0: tf.Tensor,
        gamma: tf.Tensor,
        sigma: tf.Tensor,
        s_sat: Optional[tf.Tensor],
    ) -> tf.Tensor:
        lp = prior_beta0.log_prob(beta0) + prior_sigma.log_prob(sigma)
        if num_controls > 0 and prior_gamma is not None:
            lp += tf.reduce_sum(prior_gamma.log_prob(gamma))
        if priors.use_saturation and prior_s_sat is not None and s_sat is not None:
            lp += prior_s_sat.log_prob(s_sat)
        return lp

    like_family = priors.likelihood
    if like_family not in {"normal", "student_t"}:
        raise ValueError(f"Unknown priors.likelihood={like_family!r}")

    if priors.residual_mode not in {"iid", "ar1"}:
        raise NotImplementedError(f"Unknown residual_mode={priors.residual_mode!r}")

    nu = const(priors.student_t_df)

    def _log_likelihood(
        series: tf.Tensor,
        beta_vec: tf.Tensor,
        beta0: tf.Tensor,
        gamma: tf.Tensor,
        sigma: tf.Tensor,
        phi: Optional[tf.Tensor],
    ) -> tf.Tensor:
        mean = beta0 + tf.linalg.matvec(series, beta_vec)
        if num_controls > 0:
            mean = mean + tf.linalg.matvec(Z_tf, gamma)

        sigma_safe = tf.maximum(sigma, const(SIGMA_MIN))

        if priors.residual_mode == "iid":
            if like_family == "normal":
                dist = tfd.Normal(loc=mean, scale=sigma_safe)
            else:
                dist = tfd.StudentT(df=nu, loc=mean, scale=sigma_safe)
            return tf.reduce_sum(dist.log_prob(y_tf))
        else:
            z = y_tf - mean
            if tf.size(z) == 0:
                return tf.constant(0.0, dtype=DTYPE)
            if phi is None:
                raise ValueError("phi is required when residual_mode='ar1'")
            z0 = z[:1] * tf.sqrt(const(1.0) - phi * phi)
            z1 = z[1:] - phi * z[:-1]
            zt = tf.concat([z0, z1], axis=0)
            if like_family == "normal":
                dist = tfd.Normal(loc=const(0.0), scale=sigma_safe)
            else:
                dist = tfd.StudentT(df=nu, loc=const(0.0), scale=sigma_safe)
            return tf.reduce_sum(dist.log_prob(zt))

    def _tactical_series(delta: tf.Tensor, s_param: Optional[tf.Tensor]) -> tf.Tensor:
        series = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
        if priors.use_saturation:
            if s_param is None:
                raise ValueError("saturation parameter must be provided when use_saturation=True")
            series = _saturate_log1p(series, s_param)
        return series

    hs_global = const(priors.hs_global_scale)
    hs_slab_scale = const(priors.hs_slab_scale)
    hs_slab_df = const(priors.hs_slab_df)
    if priors.hs_slab_df > 2.0:
        c2 = hs_slab_scale**2 * (hs_slab_df / (hs_slab_df - const(2.0)))
    else:
        c2 = hs_slab_scale**2

    half_cauchy_global: Optional[tfd.Distribution] = None
    half_cauchy_local: Optional[tfd.Distribution] = None
    if sparsity == "horseshoe":
        half_cauchy_global = tfd.HalfCauchy(loc=const(0.0), scale=hs_global)
        half_cauchy_local = tfd.HalfCauchy(loc=const(0.0), scale=const(1.0))

    param_spec.extend(decay_specs)
    param_spec.append(ParamSpec("sigma", (), tfb.Softplus(), const(1.0)))

    decay_param_names = [spec.name for spec in decay_specs]
    spec_names = [spec.name for spec in param_spec]

    def target_log_prob(*params: tf.Tensor) -> tf.Tensor:
        tensors = {name: tensor for name, tensor in zip(spec_names, params)}
        beta0 = tensors["beta0"]
        gamma = tensors["gamma"]
        sigma = tensors["sigma"]
        s_sat = tensors.get("s_sat")
        phi = tensors.get("phi")
        decay_args = [tensors[name] for name in decay_param_names]
        delta = delta_from_args(*decay_args)
        tactical_series = _tactical_series(delta, s_sat)
        series_beta = tf.linalg.matmul(tactical_series, M_leaf_to_beta)
        eta_beta = tensors[eta_beta_name]
        beta_vec = tf.exp(eta_beta)
        ll = _log_likelihood(series_beta, beta_vec, beta0, gamma, sigma, phi)

        lp = _log_prior_common(beta0, gamma, sigma, s_sat)
        if priors.residual_mode == "ar1":
            if phi is None:
                raise ValueError("phi is required when residual_mode='ar1'")
            lp += prior_phi.log_prob(phi)
        if pool_parent is None:
            lp += tf.reduce_sum(prior_eta.log_prob(eta_beta))
        else:
            if eta_parent_name is None or child_to_parent is None:
                raise ValueError("Pooling configuration requires a parent level")
            eta_parent = tensors[eta_parent_name]
            parent_vals = tf.gather(eta_parent, child_to_parent)
            lp += tf.reduce_sum(prior_eta.log_prob(eta_parent))
            if sparsity == "horseshoe":
                assert half_cauchy_global is not None and half_cauchy_local is not None
                tau0 = tensors["tau0"]
                lambda_local = tensors["lambda_local"]
                deviation = eta_beta - parent_vals
                lam2 = lambda_local**2
                tau0_sq = tau0**2
                shrink = (c2 * lam2) / (c2 + tau0_sq * lam2)
                sd = tf.sqrt(tf.maximum(tau0_sq * shrink, const(1e-12)))
                lp += tf.reduce_sum(
                    tfd.Normal(loc=const(0.0), scale=sd).log_prob(deviation)
                )
                lp += half_cauchy_global.log_prob(tau0)
                lp += tf.reduce_sum(half_cauchy_local.log_prob(lambda_local))
            else:
                tau_beta = tensors["tau_beta"]
                lp += prior_tau.log_prob(tau_beta)
                lp += tf.reduce_sum(
                    tfd.Normal(loc=parent_vals, scale=tau_beta).log_prob(eta_beta)
                )

        lp += log_prior_decay(*decay_args)
        return ll + lp

    dims = {
        "T": T_,
        "N_t": num_tacticals,
        "J": num_controls,
        "mode": decay_mode,
        "residual_mode": priors.residual_mode,
        "beta_structure": beta_structure,
        "use_saturation": priors.use_saturation,
        "tactical_rescale": np.asarray(tactical_rescale, dtype=float),
        "y_mean": y_mean,
        "levels": list(hierarchy.levels),
        "level_sizes": dict(level_sizes),
        "leaf_level": leaf_level,
        "beta_level": beta_level,
        "pool_parent_level": pool_parent,
    }
    if beta_structure == "tactical_hier":
        dims["sparsity"] = sparsity

    if C is not None:
        dims["C"] = C
    if P is not None:
        dims["P"] = P

    return target_log_prob, dims, param_spec


__all__ = ["ParamSpec", "TargetLogProbFn", "make_target_log_prob_fn"]
