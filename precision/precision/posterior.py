"""Target log posterior construction for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .adstock import DTYPE, adstock_geometric_tf
from .hierarchy import Hierarchy
from .priors import Priors


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


def make_target_log_prob_fn(
    y: np.ndarray,
    U_tactical: np.ndarray,
    Z_controls: Optional[np.ndarray],
    hierarchy: Hierarchy,
    *,
    normalize_adstock: bool = True,
    priors: Priors = Priors(),
) -> Tuple[TargetLogProbFn, Dict[str, int], List[ParamSpec]]:
    """Create a TensorFlow Probability target log-probability function."""

    const = lambda value: tf.constant(value, dtype=DTYPE)

    y_tf = tf.convert_to_tensor(y, dtype=DTYPE)
    U_tf = tf.convert_to_tensor(U_tactical, dtype=DTYPE)
    T_, num_tacticals = U_tactical.shape

    if Z_controls is None:
        Z_tf = tf.zeros((T_, 0), dtype=DTYPE)
        num_controls = 0
    else:
        Z_tf = tf.convert_to_tensor(Z_controls, dtype=DTYPE)
        num_controls = int(Z_controls.shape[1])

    C = int(hierarchy.num_channels)
    P = int(hierarchy.num_platforms)

    M_tc_tf = tf.convert_to_tensor(hierarchy.M_tc, dtype=DTYPE)
    M_tp_tf = tf.convert_to_tensor(hierarchy.M_tp, dtype=DTYPE)
    t_to_p = tf.convert_to_tensor(hierarchy.t_to_p, dtype=tf.int32)
    p_to_c = tf.convert_to_tensor(hierarchy.p_to_c, dtype=tf.int32)
    t_to_c = tf.gather(p_to_c, t_to_p)

    prior_beta0 = tfd.Normal(loc=const(0.0), scale=const(priors.beta0_sd))
    prior_beta_ch = tfd.Normal(loc=const(0.0), scale=const(priors.beta_sd))
    prior_gamma = (
        tfd.Normal(loc=const(0.0), scale=const(priors.gamma_sd))
        if num_controls > 0
        else None
    )
    prior_sigma = tfd.HalfNormal(scale=const(priors.sigma_sd))

    beta_structure = priors.beta_structure
    sparsity = priors.sparsity_prior
    if beta_structure not in {"channel", "platform_hier", "tactical_hier"}:
        raise ValueError(f"Unknown priors.beta_structure={beta_structure!r}")
    if sparsity not in {"none", "horseshoe", "dl"}:
        raise ValueError(f"Unknown priors.sparsity_prior={sparsity!r}")

    if beta_structure != "tactical_hier" and sparsity != "none":
        raise ValueError("Sparsity priors are only supported with tactical_hier betas")
    if sparsity == "horseshoe" and beta_structure != "tactical_hier":
        raise ValueError("Horseshoe prior requires tactical_hier beta structure")

    param_spec: List[ParamSpec] = []
    param_spec.append(ParamSpec("beta0", (), tfb.Identity(), tf.zeros([], DTYPE)))
    param_spec.append(
        ParamSpec("beta_channel", (C,), tfb.Identity(), tf.zeros([C], DTYPE))
    )
    param_spec.append(
        ParamSpec("gamma", (num_controls,), tfb.Identity(), tf.zeros([num_controls], DTYPE))
    )

    # Beta hierarchy parameters.
    if beta_structure == "platform_hier":
        param_spec.append(
            ParamSpec(
                "tau_beta",
                (),
                tfb.Softplus(),
                const(priors.beta_pool_sd),
            )
        )
        param_spec.append(
            ParamSpec("beta_platform", (P,), tfb.Identity(), tf.zeros([P], DTYPE))
        )
    elif beta_structure == "tactical_hier":
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
            ParamSpec("beta_tactical", (num_tacticals,), tfb.Identity(), tf.zeros([num_tacticals], DTYPE))
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
                    (num_tacticals,),
                    tfb.Softplus(),
                    tf.ones([num_tacticals], dtype=DTYPE),
                )
            )

    # Decay configuration.
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
            return tf.pow(const(2.0), -1.0 / tf.maximum(h, const(1e-6)))

        def log_prior_decay(log_h: tf.Tensor) -> tf.Tensor:
            return tf.reduce_sum(prior_log_h.log_prob(log_h))

    elif decay_mode == "hier_logit":
        prior_mu_c = tfd.Normal(loc=const(priors.hier_mu0), scale=const(priors.hier_mu0_sd))
        prior_tau_c = tfd.HalfNormal(scale=const(priors.hier_tau_sd))

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
            del mu_c  # unused in reconstruction
            del log_tau_c
            return tf.math.sigmoid(logit_delta)

        def log_prior_decay(logit_delta: tf.Tensor, mu_c: tf.Tensor, log_tau_c: tf.Tensor) -> tf.Tensor:
            tau_c = tf.math.softplus(log_tau_c)
            mu_i = tf.gather(mu_c, t_to_c)
            tau_i = tf.gather(tau_c, t_to_c)
            log_p = tf.reduce_sum(prior_mu_c.log_prob(mu_c))
            log_p += tf.reduce_sum(prior_tau_c.log_prob(tau_c))
            log_p += tf.reduce_sum(tf.math.log_sigmoid(log_tau_c))
            log_p += tf.reduce_sum(tfd.Normal(mu_i, tau_i).log_prob(logit_delta))
            return log_p

    else:
        raise ValueError(f"Unknown priors.decay_mode={decay_mode!r}")

    param_spec.extend(decay_specs)
    sigma_spec = ParamSpec("sigma", (), tfb.Softplus(), const(1.0))
    param_spec.append(sigma_spec)

    def _log_prior_common(
        beta0: tf.Tensor,
        beta_channel: tf.Tensor,
        gamma: tf.Tensor,
        sigma: tf.Tensor,
    ) -> tf.Tensor:
        lp = (
            prior_beta0.log_prob(beta0)
            + tf.reduce_sum(prior_beta_ch.log_prob(beta_channel))
            + prior_sigma.log_prob(sigma)
        )
        if num_controls > 0 and prior_gamma is not None:
            lp += tf.reduce_sum(prior_gamma.log_prob(gamma))
        return lp

    def _likelihood(series: tf.Tensor, beta_vec: tf.Tensor, beta0: tf.Tensor, gamma: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
        mean = beta0 + tf.linalg.matvec(series, beta_vec)
        if num_controls > 0:
            mean = mean + tf.linalg.matvec(Z_tf, gamma)
        return tf.reduce_sum(tfd.Normal(loc=mean, scale=sigma).log_prob(y_tf))

    # Channel-only structure -------------------------------------------------
    if beta_structure == "channel":

        def target_log_prob(
            beta0: tf.Tensor,
            beta_channel: tf.Tensor,
            gamma: tf.Tensor,
            *decay_and_sigma: tf.Tensor,
        ) -> tf.Tensor:
            sigma = decay_and_sigma[-1]
            decay_args = decay_and_sigma[:-1]
            delta = delta_from_args(*decay_args)
            adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
            channel_series = tf.linalg.matmul(adstocked, M_tc_tf)
            ll = _likelihood(channel_series, beta_channel, beta0, gamma, sigma)
            lp = _log_prior_common(beta0, beta_channel, gamma, sigma)
            lp += log_prior_decay(*decay_args)
            return ll + lp

        dims = {
            "T": T_,
            "N_t": num_tacticals,
            "C": C,
            "P": P,
            "J": num_controls,
            "mode": decay_mode,
            "beta_structure": "channel",
        }
        return target_log_prob, dims, param_spec

    # Platform hierarchy -----------------------------------------------------
    if beta_structure == "platform_hier":
        prior_tau = tfd.HalfNormal(scale=const(priors.beta_pool_sd))

        def target_log_prob(
            beta0: tf.Tensor,
            beta_channel: tf.Tensor,
            gamma: tf.Tensor,
            tau_beta: tf.Tensor,
            beta_platform: tf.Tensor,
            *decay_and_sigma: tf.Tensor,
        ) -> tf.Tensor:
            sigma = decay_and_sigma[-1]
            decay_args = decay_and_sigma[:-1]
            delta = delta_from_args(*decay_args)
            adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
            platform_series = tf.linalg.matmul(adstocked, M_tp_tf)
            ll = _likelihood(platform_series, beta_platform, beta0, gamma, sigma)

            beta_ch_for_p = tf.gather(beta_channel, p_to_c)
            lp = _log_prior_common(beta0, beta_channel, gamma, sigma)
            lp += prior_tau.log_prob(tau_beta)
            lp += tf.reduce_sum(
                tfd.Normal(loc=beta_ch_for_p, scale=tau_beta).log_prob(beta_platform)
            )
            lp += log_prior_decay(*decay_args)
            return ll + lp

        dims = {
            "T": T_,
            "N_t": num_tacticals,
            "C": C,
            "P": P,
            "J": num_controls,
            "mode": decay_mode,
            "beta_structure": "platform_hier",
        }
        return target_log_prob, dims, param_spec

    # Tactical hierarchy -----------------------------------------------------
    prior_tau = tfd.HalfNormal(scale=const(priors.beta_pool_sd))
    hs_global = const(priors.hs_global_scale)
    hs_slab_scale = const(priors.hs_slab_scale)
    hs_slab_df = const(priors.hs_slab_df)
    if priors.hs_slab_df > 2.0:
        c2 = hs_slab_scale ** 2 * (hs_slab_df / (hs_slab_df - const(2.0)))
    else:
        c2 = hs_slab_scale ** 2

    if sparsity == "horseshoe":
        half_cauchy_global = tfd.HalfCauchy(scale=hs_global)
        half_cauchy_local = tfd.HalfCauchy(scale=const(1.0))

        def target_log_prob(
            beta0: tf.Tensor,
            beta_channel: tf.Tensor,
            gamma: tf.Tensor,
            beta_tactical: tf.Tensor,
            tau0: tf.Tensor,
            lambda_local: tf.Tensor,
            *decay_and_sigma: tf.Tensor,
        ) -> tf.Tensor:
            sigma = decay_and_sigma[-1]
            decay_args = decay_and_sigma[:-1]
            delta = delta_from_args(*decay_args)
            adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
            ll = _likelihood(adstocked, beta_tactical, beta0, gamma, sigma)

            beta_ch_for_t = tf.gather(beta_channel, t_to_c)
            deviation = beta_tactical - beta_ch_for_t
            lam2 = lambda_local ** 2
            tau0_sq = tau0 ** 2
            shrink = (c2 * lam2) / (c2 + tau0_sq * lam2)
            sd = tf.sqrt(tf.maximum(tau0_sq * shrink, const(1e-12)))

            lp = _log_prior_common(beta0, beta_channel, gamma, sigma)
            lp += tf.reduce_sum(tfd.Normal(loc=const(0.0), scale=sd).log_prob(deviation))
            lp += half_cauchy_global.log_prob(tau0)
            lp += tf.reduce_sum(half_cauchy_local.log_prob(lambda_local))
            lp += log_prior_decay(*decay_args)
            return ll + lp

        dims = {
            "T": T_,
            "N_t": num_tacticals,
            "C": C,
            "P": P,
            "J": num_controls,
            "mode": decay_mode,
            "beta_structure": "tactical_hier",
            "sparsity": "horseshoe",
        }
        return target_log_prob, dims, param_spec

    # Tactical hierarchy without sparsity -----------------------------------
    def target_log_prob(
        beta0: tf.Tensor,
        beta_channel: tf.Tensor,
        gamma: tf.Tensor,
        tau_beta: tf.Tensor,
        beta_tactical: tf.Tensor,
        *decay_and_sigma: tf.Tensor,
    ) -> tf.Tensor:
        sigma = decay_and_sigma[-1]
        decay_args = decay_and_sigma[:-1]
        delta = delta_from_args(*decay_args)
        adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
        ll = _likelihood(adstocked, beta_tactical, beta0, gamma, sigma)

        beta_ch_for_t = tf.gather(beta_channel, t_to_c)
        deviation = beta_tactical - beta_ch_for_t
        lp = _log_prior_common(beta0, beta_channel, gamma, sigma)
        lp += prior_tau.log_prob(tau_beta)
        lp += tf.reduce_sum(
            tfd.Normal(loc=const(0.0), scale=tau_beta).log_prob(deviation)
        )
        lp += log_prior_decay(*decay_args)
        return ll + lp

    dims = {
        "T": T_,
        "N_t": num_tacticals,
        "C": C,
        "P": P,
        "J": num_controls,
        "mode": decay_mode,
        "beta_structure": "tactical_hier",
        "sparsity": "none",
    }
    return target_log_prob, dims, param_spec


__all__ = ["ParamSpec", "TargetLogProbFn", "make_target_log_prob_fn"]
