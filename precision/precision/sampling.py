"""Posterior sampling utilities for the Precision MMM package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .adstock import DTYPE
from .posterior import ParamSpec, TargetLogProbFn


tfb = tfp.bijectors


def _stack_last_two(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array
    return array.reshape((-1, *array.shape[2:]))


def _softplus_numpy(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


@dataclass
class PosteriorSamples:
    """Posterior draws in constrained space with derived decay parameters."""

    beta0: np.ndarray
    beta_channel: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    sigma: np.ndarray
    half_life: np.ndarray | None = None
    logit_delta: np.ndarray | None = None
    mu_c: np.ndarray | None = None
    tau_c: np.ndarray | None = None
    beta_platform: np.ndarray | None = None
    beta_tactical: np.ndarray | None = None
    tau_beta: np.ndarray | None = None
    tau0: np.ndarray | None = None
    lambda_local: np.ndarray | None = None
    s_sat: np.ndarray | None = None

    def stack_chains(self) -> "PosteriorSamples":
        """Return a copy with chain and sample dimensions flattened."""

        return PosteriorSamples(
            beta0=_stack_last_two(self.beta0),
            beta_channel=_stack_last_two(self.beta_channel),
            gamma=_stack_last_two(self.gamma),
            delta=_stack_last_two(self.delta),
            sigma=_stack_last_two(self.sigma),
            half_life=None if self.half_life is None else _stack_last_two(self.half_life),
            logit_delta=None
            if self.logit_delta is None
            else _stack_last_two(self.logit_delta),
            mu_c=None if self.mu_c is None else _stack_last_two(self.mu_c),
            tau_c=None if self.tau_c is None else _stack_last_two(self.tau_c),
            beta_platform=None
            if self.beta_platform is None
            else _stack_last_two(self.beta_platform),
            beta_tactical=None
            if self.beta_tactical is None
            else _stack_last_two(self.beta_tactical),
            tau_beta=None if self.tau_beta is None else _stack_last_two(self.tau_beta),
            tau0=None if self.tau0 is None else _stack_last_two(self.tau0),
            lambda_local=None
            if self.lambda_local is None
            else _stack_last_two(self.lambda_local),
            s_sat=None if self.s_sat is None else _stack_last_two(self.s_sat),
        )


def run_nuts(
    target_log_prob_fn: TargetLogProbFn,
    dims: Dict[str, int],
    param_spec: List[ParamSpec],
    *,
    num_chains: int = 4,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    init_step_size: float = 0.1,
    seed: int | None = 42,
) -> PosteriorSamples:
    """Run the No-U-Turn Sampler and return constrained samples with delta."""

    tf.random.set_seed(seed)

    bijectors = [spec.bijector for spec in param_spec]
    init_states = []
    for spec in param_spec:
        if spec.shape:
            target_shape = (num_chains, *spec.shape)
        else:
            target_shape = (num_chains,)
        init_states.append(tf.broadcast_to(spec.init, target_shape))

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
    def _sample():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin,
            current_state=init_states,
            kernel=adapt,
            trace_fn=None,
            return_final_kernel_results=False,
        )

    samples_constrained = list(_sample())

    name_to_tensor = {
        spec.name: tensor.numpy() for spec, tensor in zip(param_spec, samples_constrained)
    }

    mode = dims.get("mode", "beta")

    if mode == "beta":
        delta = name_to_tensor["delta"]
        extras = dict(half_life=None, logit_delta=None, mu_c=None, tau_c=None)
    elif mode == "half_life":
        log_h = name_to_tensor["log_h"]
        h = np.exp(log_h)
        delta = np.power(2.0, -1.0 / np.maximum(h, 1e-6))
        extras = dict(half_life=h, logit_delta=None, mu_c=None, tau_c=None)
    elif mode == "hier_logit":
        logit_delta = name_to_tensor["logit_delta"]
        delta = 1.0 / (1.0 + np.exp(-logit_delta))
        mu_c = name_to_tensor["mu_c"]
        log_tau_c = name_to_tensor["log_tau_c"]
        tau_c = _softplus_numpy(log_tau_c)
        extras = dict(half_life=None, logit_delta=logit_delta, mu_c=mu_c, tau_c=tau_c)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    beta_platform = name_to_tensor.get("beta_platform")
    beta_tactical = name_to_tensor.get("beta_tactical")
    tau_beta = name_to_tensor.get("tau_beta")
    tau0 = name_to_tensor.get("tau0")
    lambda_local = name_to_tensor.get("lambda_local")
    s_sat = name_to_tensor.get("s_sat")

    return PosteriorSamples(
        beta0=name_to_tensor["beta0"],
        beta_channel=name_to_tensor["beta_channel"],
        gamma=name_to_tensor.get("gamma", np.zeros((num_samples, num_chains, 0))),
        delta=delta,
        sigma=name_to_tensor["sigma"],
        half_life=extras.get("half_life"),
        logit_delta=extras.get("logit_delta"),
        mu_c=extras.get("mu_c"),
        tau_c=extras.get("tau_c"),
        beta_platform=beta_platform,
        beta_tactical=beta_tactical,
        tau_beta=tau_beta,
        tau0=tau0,
        lambda_local=lambda_local,
        s_sat=s_sat,
    )


__all__ = ["PosteriorSamples", "run_nuts"]
