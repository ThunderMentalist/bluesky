"""Target log posterior construction for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from .adstock import DTYPE, adstock_geometric_tf
from .hierarchy import Hierarchy
from .priors import Priors


TargetLogProbFn = Callable[..., tf.Tensor]


@dataclass
class ParamSpec:
    """Specification for a model parameter used to construct NUTS kernels."""

    name: str
    shape: Tuple[int, ...]
    bijector: tfp.bijectors.Bijector
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
    """Create a TensorFlow Probability target log-probability function.

    Parameters
    ----------
    y:
        Array of observed responses with shape ``[T]``.
    U_tactical:
        Tactical design matrix with shape ``[T, N_t]``.
    Z_controls:
        Optional control covariate matrix with shape ``[T, J]``.
    hierarchy:
        ``Hierarchy`` instance containing aggregation matrices.
    normalize_adstock:
        If ``True`` (default), apply normalized adstock when constructing the
        likelihood.
    priors:
        Prior configuration, including decay-mode specific hyperparameters.

    Returns
    -------
    target_fn, dims, param_spec:
        ``target_fn`` computes the joint log density, ``dims`` summarises
        dimensionalities, and ``param_spec`` provides bijectors and initial
        values for NUTS initialisation.
    """

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
    M_tc_tf = tf.convert_to_tensor(hierarchy.M_tc, dtype=DTYPE)

    tfd = tfp.distributions
    tfb = tfp.bijectors

    prior_beta0 = tfd.Normal(loc=DTYPE(0.0), scale=DTYPE(priors.beta0_sd))
    prior_beta = tfd.Normal(loc=DTYPE(0.0), scale=DTYPE(priors.beta_sd))
    prior_gamma = (
        tfd.Normal(loc=DTYPE(0.0), scale=DTYPE(priors.gamma_sd))
        if num_controls > 0
        else None
    )
    prior_sigma = tfd.HalfNormal(scale=DTYPE(priors.sigma_sd))

    decay_mode = priors.decay_mode

    def _log_prior_common(
        beta0: tf.Tensor,
        beta_channel: tf.Tensor,
        gamma: tf.Tensor,
        sigma: tf.Tensor,
    ) -> tf.Tensor:
        log_p = (
            prior_beta0.log_prob(beta0)
            + tf.reduce_sum(prior_beta.log_prob(beta_channel))
            + prior_sigma.log_prob(sigma)
        )
        if num_controls > 0 and prior_gamma is not None:
            log_p += tf.reduce_sum(prior_gamma.log_prob(gamma))
        return log_p

    param_spec: List[ParamSpec] = []

    # Core regression parameters.
    param_spec.append(ParamSpec("beta0", (), tfb.Identity(), tf.zeros([], DTYPE)))
    param_spec.append(
        ParamSpec("beta_channel", (C,), tfb.Identity(), tf.zeros([C], DTYPE))
    )
    param_spec.append(
        ParamSpec("gamma", (num_controls,), tfb.Identity(), tf.zeros([num_controls], DTYPE))
    )
    sigma_spec = ParamSpec("sigma", (), tfb.Exp(), tf.zeros([], DTYPE))

    if decay_mode == "beta":
        prior_delta = tfd.Beta(
            concentration1=DTYPE(priors.beta_alpha),
            concentration0=DTYPE(priors.beta_beta),
        )
        param_spec.append(
            ParamSpec(
                "delta",
                (num_tacticals,),
                tfb.Sigmoid(),
                tf.fill([num_tacticals], DTYPE(0.30)),
            )
        )
        param_spec.append(sigma_spec)

        def target_log_prob(
            beta0: tf.Tensor,
            beta_channel: tf.Tensor,
            gamma: tf.Tensor,
            delta: tf.Tensor,
            sigma: tf.Tensor,
        ) -> tf.Tensor:
            adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
            channel_series = tf.linalg.matmul(adstocked, M_tc_tf)
            mean = beta0 + tf.linalg.matvec(channel_series, beta_channel)
            if num_controls > 0:
                mean = mean + tf.linalg.matvec(Z_tf, gamma)
            ll = tf.reduce_sum(tfd.Normal(loc=mean, scale=sigma).log_prob(y_tf))
            log_prior = _log_prior_common(beta0, beta_channel, gamma, sigma)
            log_prior += tf.reduce_sum(prior_delta.log_prob(delta))
            return ll + log_prior

        dims = {
            "T": T_,
            "N_t": num_tacticals,
            "C": C,
            "J": num_controls,
            "mode": "beta",
        }
        return target_log_prob, dims, param_spec

    if decay_mode == "half_life":
        prior_log_h = tfd.Normal(
            loc=DTYPE(priors.half_life_log_mu),
            scale=DTYPE(priors.half_life_log_sd),
        )
        param_spec.append(
            ParamSpec(
                "log_h",
                (num_tacticals,),
                tfb.Identity(),
                tf.fill([num_tacticals], DTYPE(priors.half_life_log_mu)),
            )
        )
        param_spec.append(sigma_spec)

        def target_log_prob(
            beta0: tf.Tensor,
            beta_channel: tf.Tensor,
            gamma: tf.Tensor,
            log_h: tf.Tensor,
            sigma: tf.Tensor,
        ) -> tf.Tensor:
            h = tf.exp(log_h)
            delta = tf.pow(DTYPE(2.0), -1.0 / tf.maximum(h, DTYPE(1e-6)))
            adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
            channel_series = tf.linalg.matmul(adstocked, M_tc_tf)
            mean = beta0 + tf.linalg.matvec(channel_series, beta_channel)
            if num_controls > 0:
                mean = mean + tf.linalg.matvec(Z_tf, gamma)
            ll = tf.reduce_sum(tfd.Normal(loc=mean, scale=sigma).log_prob(y_tf))
            log_prior = _log_prior_common(beta0, beta_channel, gamma, sigma)
            log_prior += tf.reduce_sum(prior_log_h.log_prob(log_h))
            return ll + log_prior

        dims = {
            "T": T_,
            "N_t": num_tacticals,
            "C": C,
            "J": num_controls,
            "mode": "half_life",
        }
        return target_log_prob, dims, param_spec

    if decay_mode == "hier_logit":
        t_to_p = tf.convert_to_tensor(hierarchy.t_to_p, dtype=tf.int32)
        p_to_c = tf.convert_to_tensor(hierarchy.p_to_c, dtype=tf.int32)
        t_to_c = tf.gather(p_to_c, t_to_p)

        prior_mu_c = tfd.Normal(loc=DTYPE(priors.hier_mu0), scale=DTYPE(priors.hier_mu0_sd))
        prior_tau_c = tfd.HalfNormal(scale=DTYPE(priors.hier_tau_sd))

        param_spec.append(
            ParamSpec("logit_delta", (num_tacticals,), tfb.Identity(), tf.zeros([num_tacticals], DTYPE))
        )
        param_spec.append(
            ParamSpec("mu_c", (C,), tfb.Identity(), tf.fill([C], DTYPE(priors.hier_mu0)))
        )
        param_spec.append(
            ParamSpec(
                "log_tau_c",
                (C,),
                tfb.Identity(),
                tf.fill(
                    [C],
                    tfp.math.softplus_inverse(DTYPE(0.5)),
                ),
            )
        )
        param_spec.append(sigma_spec)

        def target_log_prob(
            beta0: tf.Tensor,
            beta_channel: tf.Tensor,
            gamma: tf.Tensor,
            logit_delta: tf.Tensor,
            mu_c: tf.Tensor,
            log_tau_c: tf.Tensor,
            sigma: tf.Tensor,
        ) -> tf.Tensor:
            delta = tf.math.sigmoid(logit_delta)
            adstocked = adstock_geometric_tf(U_tf, delta, normalize=normalize_adstock)
            channel_series = tf.linalg.matmul(adstocked, M_tc_tf)
            mean = beta0 + tf.linalg.matvec(channel_series, beta_channel)
            if num_controls > 0:
                mean = mean + tf.linalg.matvec(Z_tf, gamma)
            ll = tf.reduce_sum(tfd.Normal(loc=mean, scale=sigma).log_prob(y_tf))

            tau_c = tf.math.softplus(log_tau_c)
            mu_i = tf.gather(mu_c, t_to_c)
            tau_i = tf.gather(tau_c, t_to_c)

            log_prior = _log_prior_common(beta0, beta_channel, gamma, sigma)
            log_prior += tf.reduce_sum(prior_mu_c.log_prob(mu_c))
            log_prior += tf.reduce_sum(prior_tau_c.log_prob(tau_c))
            log_prior += tf.reduce_sum(tf.math.log_sigmoid(log_tau_c))
            log_prior += tf.reduce_sum(tfd.Normal(mu_i, tau_i).log_prob(logit_delta))
            return ll + log_prior

        dims = {
            "T": T_,
            "N_t": num_tacticals,
            "C": C,
            "J": num_controls,
            "mode": "hier_logit",
        }
        return target_log_prob, dims, param_spec

    raise ValueError(f"Unknown priors.decay_mode={decay_mode!r}")


__all__ = ["ParamSpec", "TargetLogProbFn", "make_target_log_prob_fn"]
