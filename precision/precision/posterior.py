"""Target log posterior construction for the Precision MMM package."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .adstock import DTYPE, adstock_geometric_tf
from .hierarchy import Hierarchy
from .priors import Priors


TargetLogProbFn = Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def make_target_log_prob_fn(
    y: np.ndarray,
    U_tactical: np.ndarray,
    Z_controls: Optional[np.ndarray],
    hierarchy: Hierarchy,
    *,
    normalize_adstock: bool = False,
    priors: Priors = Priors(),
) -> Tuple[TargetLogProbFn, Dict[str, int]]:
    """Create a TensorFlow Probability target log-probability function."""

    y_tf = tf.convert_to_tensor(y, dtype=DTYPE)
    U_tf = tf.convert_to_tensor(U_tactical, dtype=DTYPE)
    T_, num_tacticals = U_tactical.shape

    if Z_controls is None:
        Z_tf = tf.zeros((T_, 0), dtype=DTYPE)
        num_controls = 0
    else:
        Z_tf = tf.convert_to_tensor(Z_controls, dtype=DTYPE)
        num_controls = int(Z_controls.shape[1])

    M_tc_tf = tf.convert_to_tensor(hierarchy.M_tc, dtype=DTYPE)

    tfd = tfp.distributions

    prior_beta0 = tfd.Normal(loc=DTYPE(0.0), scale=DTYPE(priors.beta0_sd))
    prior_beta = tfd.Normal(loc=DTYPE(0.0), scale=DTYPE(priors.beta_sd))
    prior_gamma = (
        tfd.Normal(loc=DTYPE(0.0), scale=DTYPE(priors.gamma_sd))
        if num_controls > 0
        else None
    )
    prior_sigma = tfd.HalfNormal(scale=DTYPE(priors.sigma_sd))
    prior_delta = tfd.Beta(
        concentration1=DTYPE(priors.beta_alpha),
        concentration0=DTYPE(priors.beta_beta),
    )

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

        likelihood = tfd.Normal(loc=mean, scale=sigma)
        log_likelihood = tf.reduce_sum(likelihood.log_prob(y_tf))

        log_prior = (
            prior_beta0.log_prob(beta0)
            + tf.reduce_sum(prior_beta.log_prob(beta_channel))
            + tf.reduce_sum(prior_delta.log_prob(delta))
            + prior_sigma.log_prob(sigma)
        )
        if num_controls > 0:
            log_prior += tf.reduce_sum(prior_gamma.log_prob(gamma))

        return log_likelihood + log_prior

    dims = {"T": T_, "N_t": num_tacticals, "C": hierarchy.num_channels, "J": num_controls}
    return target_log_prob, dims


__all__ = ["make_target_log_prob_fn", "TargetLogProbFn"]
