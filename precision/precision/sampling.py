"""Posterior sampling utilities for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .adstock import DTYPE
from .posterior import TargetLogProbFn


tfb = tfp.bijectors

def _build_identity_tensor(num_chains: int, *shape: int) -> tf.Tensor:
    return tf.zeros([num_chains, *shape], dtype=DTYPE)


@dataclass
class PosteriorSamples:
    """Container for posterior samples returned by :func:`run_nuts`."""

    beta0: np.ndarray
    beta_channel: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    sigma: np.ndarray

    def stack_chains(self) -> "PosteriorSamples":
        """Return a copy with chain and sample dimensions flattened."""

        def _reshape(value: np.ndarray) -> np.ndarray:
            if value.size == 0:
                return value
            return value.reshape((-1, *value.shape[2:]))

        return PosteriorSamples(
            beta0=_reshape(self.beta0),
            beta_channel=_reshape(self.beta_channel),
            gamma=_reshape(self.gamma),
            delta=_reshape(self.delta),
            sigma=_reshape(self.sigma),
        )


def run_nuts(
    target_log_prob_fn: TargetLogProbFn,
    dims: Dict[str, int],
    *,
    num_chains: int = 4,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    init_step_size: float = 0.1,
    seed: int | None = 42,
) -> PosteriorSamples:
    """Run the No-U-Turn Sampler to obtain posterior draws."""

    tf.random.set_seed(seed)
    C = int(dims["C"])
    J = int(dims["J"])
    N_t = int(dims["N_t"])

    beta0_init = tf.zeros([num_chains], dtype=DTYPE)
    beta_channel_init = tf.zeros([num_chains, C], dtype=DTYPE)
    gamma_init = _build_identity_tensor(num_chains, J)
    delta_init = tf.fill([num_chains, N_t], DTYPE(0.30))
    sigma_init = tf.fill([num_chains], DTYPE(1.0))

    bijectors: List[tfb.Bijector] = [
        tfb.Identity(),
        tfb.Identity(),
        tfb.Identity(),
        tfb.Sigmoid(),
        tfb.Exp(),
    ]

    unconstrained_state = [
        bijectors[0].inverse(beta0_init),
        bijectors[1].inverse(beta_channel_init),
        bijectors[2].inverse(gamma_init),
        bijectors[3].inverse(delta_init),
        bijectors[4].inverse(sigma_init),
    ]

    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=DTYPE(init_step_size),
        seed=seed,
    )

    transformed = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=nuts,
        bijector=bijectors,
    )

    adapt = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=transformed,
        num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=DTYPE(0.8),
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    @tf.function(autograph=False, jit_compile=False)
    def _sample_chain():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin,
            current_state=unconstrained_state,
            kernel=adapt,
            trace_fn=None,
            return_final_kernel_results=False,
        )

    unconstrained_samples = _sample_chain()
    constrained_samples = [
        bijector.forward(state) for bijector, state in zip(bijectors, unconstrained_samples)
    ]
    beta0_s, beta_channel_s, gamma_s, delta_s, sigma_s = constrained_samples

    return PosteriorSamples(
        beta0=beta0_s.numpy(),
        beta_channel=beta_channel_s.numpy(),
        gamma=gamma_s.numpy(),
        delta=delta_s.numpy(),
        sigma=sigma_s.numpy(),
    )


__all__ = ["PosteriorSamples", "run_nuts"]
