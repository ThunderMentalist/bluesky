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

    s_safe = tf.maximum(s, tf.cast(1e-9, DTYPE))
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
    """Create a TensorFlow Probability target log-probability function."""

    const = lambda value: tf.constant(value, dtype=DTYPE)

    T_, num_tacticals = U_tactical.shape
    C = int(hierarchy.num_channels)
    P = int(hierarchy.num_platforms)

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
    M_tc_tf = tf.convert_to_tensor(hierarchy.M_tc, dtype=DTYPE)
    M_tp_tf = tf.convert_to_tensor(hierarchy.M_tp, dtype=DTYPE)
    t_to_p = tf.convert_to_tensor(hierarchy.t_to_p, dtype=tf.int32)
    p_to_c = tf.convert_to_tensor(hierarchy.p_to_c, dtype=tf.int32)
    t_to_c = tf.gather(p_to_c, t_to_p)

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
    prior_eta_channel = tfd.Normal(
        loc=const(priors.lift_log_mu), scale=const(priors.lift_log_sd)
    )
    prior_tau = tfd.HalfNormal(scale=const(priors.beta_pool_sd))

    beta_structure = priors.beta_structure
    sparsity = priors.sparsity_prior
    if beta_structure not in {"channel", "platform_hier", "tactical_hier"}:
        raise ValueError(f"Unknown priors.beta_structure={beta_structure!r}")
    if sparsity not in {"none", "horseshoe"}:
        raise ValueError(f"Unknown priors.sparsity_prior={sparsity!r}")
    if beta_structure != "tactical_hier" and sparsity != "none":
        raise ValueError("Sparsity priors are only supported with tactical_hier betas")

    param_spec: List[ParamSpec] = []
    param_spec.append(ParamSpec("beta0", (), tfb.Identity(), tf.zeros([], DTYPE)))
    param_spec.append(
        ParamSpec("gamma", (num_controls,), tfb.Identity(), tf.zeros([num_controls], DTYPE))
    )
    if priors.use_saturation:
        param_spec.append(
            ParamSpec("s_sat", (), tfb.Exp(), const(priors.sat_log_mu))
        )
    param_spec.append(
        ParamSpec(
            "eta_channel",
            (C,),
            tfb.Identity(),
            tf.fill([C], const(priors.lift_log_mu)),
        )
    )

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
            ParamSpec(
                "eta_platform",
                (P,),
                tfb.Identity(),
                tf.fill([P], const(priors.lift_log_mu)),
            )
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
            ParamSpec(
                "eta_tactical",
                (num_tacticals,),
                tfb.Identity(),
                tf.fill([num_tacticals], const(priors.lift_log_mu)),
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
                    (num_tacticals,),
                    tfb.Softplus(),
                    tf.ones([num_tacticals], dtype=DTYPE),
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
            del mu_c
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
    param_spec.append(ParamSpec("sigma", (), tfb.Softplus(), const(1.0)))

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

    if priors.residual_mode != "iid":
        raise NotImplementedError("Only IID residuals are supported in this build")

    nu = const(priors.student_t_df)

    def _log_likelihood(
        series: tf.Tensor,
        beta_vec: tf.Tensor,
        beta0: tf.Tensor,
        gamma: tf.Tensor,
        sigma: tf.Tensor,
    ) -> tf.Tensor:
        mean = beta0 + tf.linalg.matvec(series, beta_vec)
        if num_controls > 0:
            mean = mean + tf.linalg.matvec(Z_tf, gamma)

        if like_family == "normal":
            dist = tfd.Normal(loc=mean, scale=sigma)
        else:
            dist = tfd.StudentT(df=nu, loc=mean, scale=sigma)
        return tf.reduce_sum(dist.log_prob(y_tf))

    def _tactical_series(delta: tf.Tensor, s_param: Optional[tf.Tensor]) -> tf.Tensor:
        series = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
        if priors.use_saturation:
            if s_param is None:
                raise ValueError("saturation parameter must be provided when use_saturation=True")
            series = _saturate_log1p(series, s_param)
        return series

    if beta_structure == "channel":

        if priors.use_saturation:

            def target_log_prob(
                beta0: tf.Tensor,
                gamma: tf.Tensor,
                s_sat: tf.Tensor,
                eta_channel: tf.Tensor,
                *decay_and_sigma: tf.Tensor,
            ) -> tf.Tensor:
                sigma = decay_and_sigma[-1]
                decay_args = decay_and_sigma[:-1]
                delta = delta_from_args(*decay_args)
                tactical_series = _tactical_series(delta, s_sat)
                channel_series = tf.linalg.matmul(tactical_series, M_tc_tf)
                beta_channel = tf.exp(eta_channel)
                ll = _log_likelihood(channel_series, beta_channel, beta0, gamma, sigma)
                lp = _log_prior_common(beta0, gamma, sigma, s_sat)
                lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
                lp += log_prior_decay(*decay_args)
                return ll + lp

        else:

            def target_log_prob(
                beta0: tf.Tensor,
                gamma: tf.Tensor,
                eta_channel: tf.Tensor,
                *decay_and_sigma: tf.Tensor,
            ) -> tf.Tensor:
                sigma = decay_and_sigma[-1]
                decay_args = decay_and_sigma[:-1]
                delta = delta_from_args(*decay_args)
                tactical_series = _tactical_series(delta, None)
                channel_series = tf.linalg.matmul(tactical_series, M_tc_tf)
                beta_channel = tf.exp(eta_channel)
                ll = _log_likelihood(channel_series, beta_channel, beta0, gamma, sigma)
                lp = _log_prior_common(beta0, gamma, sigma, None)
                lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
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
            "use_saturation": priors.use_saturation,
            "tactical_rescale": tactical_rescale.tolist(),
            "y_mean": y_mean,
        }
        return target_log_prob, dims, param_spec

    if beta_structure == "platform_hier":

        if priors.use_saturation:

            def target_log_prob(
                beta0: tf.Tensor,
                gamma: tf.Tensor,
                s_sat: tf.Tensor,
                eta_channel: tf.Tensor,
                tau_beta: tf.Tensor,
                eta_platform: tf.Tensor,
                *decay_and_sigma: tf.Tensor,
            ) -> tf.Tensor:
                sigma = decay_and_sigma[-1]
                decay_args = decay_and_sigma[:-1]
                delta = delta_from_args(*decay_args)
                tactical_series = _tactical_series(delta, s_sat)
                platform_series = tf.linalg.matmul(tactical_series, M_tp_tf)
                beta_platform = tf.exp(eta_platform)
                ll = _log_likelihood(platform_series, beta_platform, beta0, gamma, sigma)

                eta_ch_for_p = tf.gather(eta_channel, p_to_c)
                lp = _log_prior_common(beta0, gamma, sigma, s_sat)
                lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
                lp += prior_tau.log_prob(tau_beta)
                lp += tf.reduce_sum(
                    tfd.Normal(loc=eta_ch_for_p, scale=tau_beta).log_prob(eta_platform)
                )
                lp += log_prior_decay(*decay_args)
                return ll + lp

        else:

            def target_log_prob(
                beta0: tf.Tensor,
                gamma: tf.Tensor,
                eta_channel: tf.Tensor,
                tau_beta: tf.Tensor,
                eta_platform: tf.Tensor,
                *decay_and_sigma: tf.Tensor,
            ) -> tf.Tensor:
                sigma = decay_and_sigma[-1]
                decay_args = decay_and_sigma[:-1]
                delta = delta_from_args(*decay_args)
                tactical_series = _tactical_series(delta, None)
                platform_series = tf.linalg.matmul(tactical_series, M_tp_tf)
                beta_platform = tf.exp(eta_platform)
                ll = _log_likelihood(platform_series, beta_platform, beta0, gamma, sigma)

                eta_ch_for_p = tf.gather(eta_channel, p_to_c)
                lp = _log_prior_common(beta0, gamma, sigma, None)
                lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
                lp += prior_tau.log_prob(tau_beta)
                lp += tf.reduce_sum(
                    tfd.Normal(loc=eta_ch_for_p, scale=tau_beta).log_prob(eta_platform)
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
            "use_saturation": priors.use_saturation,
            "tactical_rescale": tactical_rescale.tolist(),
            "y_mean": y_mean,
        }
        return target_log_prob, dims, param_spec

    hs_global = const(priors.hs_global_scale)
    hs_slab_scale = const(priors.hs_slab_scale)
    hs_slab_df = const(priors.hs_slab_df)
    if priors.hs_slab_df > 2.0:
        c2 = hs_slab_scale**2 * (hs_slab_df / (hs_slab_df - const(2.0)))
    else:
        c2 = hs_slab_scale**2

    if sparsity == "horseshoe":
        half_cauchy_global = tfd.HalfCauchy(loc=const(0.0), scale=hs_global)
        half_cauchy_local = tfd.HalfCauchy(loc=const(0.0), scale=const(1.0))

        if priors.use_saturation:

            def target_log_prob(
                beta0: tf.Tensor,
                gamma: tf.Tensor,
                s_sat: tf.Tensor,
                eta_channel: tf.Tensor,
                eta_tactical: tf.Tensor,
                tau0: tf.Tensor,
                lambda_local: tf.Tensor,
                *decay_and_sigma: tf.Tensor,
            ) -> tf.Tensor:
                sigma = decay_and_sigma[-1]
                decay_args = decay_and_sigma[:-1]
                delta = delta_from_args(*decay_args)
                tactical_series = _tactical_series(delta, s_sat)
                beta_tactical = tf.exp(eta_tactical)
                ll = _log_likelihood(tactical_series, beta_tactical, beta0, gamma, sigma)

                eta_ch_for_t = tf.gather(eta_channel, t_to_c)
                deviation = eta_tactical - eta_ch_for_t
                lam2 = lambda_local**2
                tau0_sq = tau0**2
                shrink = (c2 * lam2) / (c2 + tau0_sq * lam2)
                sd = tf.sqrt(tf.maximum(tau0_sq * shrink, const(1e-12)))

                lp = _log_prior_common(beta0, gamma, sigma, s_sat)
                lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
                lp += tf.reduce_sum(tfd.Normal(loc=const(0.0), scale=sd).log_prob(deviation))
                lp += half_cauchy_global.log_prob(tau0)
                lp += tf.reduce_sum(half_cauchy_local.log_prob(lambda_local))
                lp += log_prior_decay(*decay_args)
                return ll + lp

        else:

            def target_log_prob(
                beta0: tf.Tensor,
                gamma: tf.Tensor,
                eta_channel: tf.Tensor,
                eta_tactical: tf.Tensor,
                tau0: tf.Tensor,
                lambda_local: tf.Tensor,
                *decay_and_sigma: tf.Tensor,
            ) -> tf.Tensor:
                sigma = decay_and_sigma[-1]
                decay_args = decay_and_sigma[:-1]
                delta = delta_from_args(*decay_args)
                tactical_series = _tactical_series(delta, None)
                beta_tactical = tf.exp(eta_tactical)
                ll = _log_likelihood(tactical_series, beta_tactical, beta0, gamma, sigma)

                eta_ch_for_t = tf.gather(eta_channel, t_to_c)
                deviation = eta_tactical - eta_ch_for_t
                lam2 = lambda_local**2
                tau0_sq = tau0**2
                shrink = (c2 * lam2) / (c2 + tau0_sq * lam2)
                sd = tf.sqrt(tf.maximum(tau0_sq * shrink, const(1e-12)))

                lp = _log_prior_common(beta0, gamma, sigma, None)
                lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
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
            "use_saturation": priors.use_saturation,
            "tactical_rescale": tactical_rescale.tolist(),
            "y_mean": y_mean,
        }
        return target_log_prob, dims, param_spec

    if priors.use_saturation:

        def target_log_prob(
            beta0: tf.Tensor,
            gamma: tf.Tensor,
            s_sat: tf.Tensor,
            eta_channel: tf.Tensor,
            tau_beta: tf.Tensor,
            eta_tactical: tf.Tensor,
            *decay_and_sigma: tf.Tensor,
        ) -> tf.Tensor:
            sigma = decay_and_sigma[-1]
            decay_args = decay_and_sigma[:-1]
            delta = delta_from_args(*decay_args)
            tactical_series = _tactical_series(delta, s_sat)
            beta_tactical = tf.exp(eta_tactical)
            ll = _log_likelihood(tactical_series, beta_tactical, beta0, gamma, sigma)

            eta_ch_for_t = tf.gather(eta_channel, t_to_c)
            lp = _log_prior_common(beta0, gamma, sigma, s_sat)
            lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
            lp += prior_tau.log_prob(tau_beta)
            lp += tf.reduce_sum(
                tfd.Normal(loc=eta_ch_for_t, scale=tau_beta).log_prob(eta_tactical)
            )
            lp += log_prior_decay(*decay_args)
            return ll + lp

    else:

        def target_log_prob(
            beta0: tf.Tensor,
            gamma: tf.Tensor,
            eta_channel: tf.Tensor,
            tau_beta: tf.Tensor,
            eta_tactical: tf.Tensor,
            *decay_and_sigma: tf.Tensor,
        ) -> tf.Tensor:
            sigma = decay_and_sigma[-1]
            decay_args = decay_and_sigma[:-1]
            delta = delta_from_args(*decay_args)
            tactical_series = _tactical_series(delta, None)
            beta_tactical = tf.exp(eta_tactical)
            ll = _log_likelihood(tactical_series, beta_tactical, beta0, gamma, sigma)

            eta_ch_for_t = tf.gather(eta_channel, t_to_c)
            lp = _log_prior_common(beta0, gamma, sigma, None)
            lp += tf.reduce_sum(prior_eta_channel.log_prob(eta_channel))
            lp += prior_tau.log_prob(tau_beta)
            lp += tf.reduce_sum(
                tfd.Normal(loc=eta_ch_for_t, scale=tau_beta).log_prob(eta_tactical)
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
        "use_saturation": priors.use_saturation,
        "tactical_rescale": tactical_rescale.tolist(),
        "y_mean": y_mean,
    }
    return target_log_prob, dims, param_spec


__all__ = ["ParamSpec", "TargetLogProbFn", "make_target_log_prob_fn"]
