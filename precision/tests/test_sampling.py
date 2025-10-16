import sys
import types

import numpy as np
import tensorflow as tf

# TensorFlow Probability requires tf_keras; provide a lightweight stub.
sys.modules.setdefault("tf_keras", types.ModuleType("tf_keras"))
import tensorflow_probability as tfp

from precision.precision.posterior import ParamSpec
from precision.precision.sampling import (
    PosteriorSamples,
    _softplus_numpy,
    _stack_last_two,
    run_nuts,
)


def test_stack_last_two_handles_shapes():
    empty = np.zeros((0, 0, 0))
    assert _stack_last_two(empty).shape == (0, 0, 0)

    array = np.arange(12).reshape(2, 3, 2)
    stacked = _stack_last_two(array)
    assert stacked.shape == (6, 2)


def test_softplus_numpy_matches_tensorflow():
    x = np.array([-5.0, 0.0, 2.5])
    np.testing.assert_allclose(_softplus_numpy(x), tf.nn.softplus(x).numpy())


def test_posterior_samples_stack_chains():
    samples = PosteriorSamples(
        beta0=np.zeros((2, 3, 1)),
        beta_channel=np.zeros((2, 3, 2)),
        gamma=np.zeros((2, 3, 1)),
        delta=np.zeros((2, 3, 2)),
        sigma=np.ones((2, 3, 1)),
    )
    stacked = samples.stack_chains()
    assert stacked.beta0.shape == (6, 1)
    assert stacked.beta_channel.shape == (6, 2)


def test_run_nuts_simple_model(monkeypatch):
    tfd = tfp.distributions
    tfb = tfp.bijectors

    import precision.precision.sampling as sampling_module

    monkeypatch.setattr(sampling_module, "DTYPE", tf.float64, raising=False)

    class FakeKernel:
        def __init__(self, **kwargs):
            self.parameters = {"target_log_prob_fn": kwargs.get("target_log_prob_fn")}

        def copy(self, **kwargs):
            new = FakeKernel()
            new.parameters = {**self.parameters, **kwargs}
            return new

    monkeypatch.setattr(sampling_module.tfp.mcmc, "NoUTurnSampler", FakeKernel)
    monkeypatch.setattr(
        sampling_module.tfp.mcmc,
        "DualAveragingStepSizeAdaptation",
        lambda inner_kernel, **kwargs: inner_kernel,
    )

    class FakeTransformedKernel:
        def __init__(self, inner_kernel, bijector):
            self.inner_kernel = inner_kernel
            self.bijector = bijector

        @property
        def parameters(self):
            return self.inner_kernel.parameters

        def copy(self, **kwargs):
            return FakeTransformedKernel(self.inner_kernel.copy(**kwargs), self.bijector)

    monkeypatch.setattr(
        sampling_module.tfp.mcmc,
        "TransformedTransitionKernel",
        FakeTransformedKernel,
    )

    def fake_sample_chain(
        num_results,
        num_burnin_steps,
        current_state,
        kernel,
        trace_fn,
        return_final_kernel_results,
        seed=None,
    ):
        outputs = []
        for state in current_state:
            outputs.append(tf.zeros((num_results, 1, *state.shape[1:]), dtype=tf.float64))
        return outputs

    monkeypatch.setattr(sampling_module.tfp.mcmc, "sample_chain", fake_sample_chain)

    def target_log_prob_fn(beta0, beta_channel, delta, sigma, eta_channel):
        dist = tfd.Normal(loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64))
        logp = tf.reduce_sum(dist.log_prob(beta0))
        logp += tf.reduce_sum(dist.log_prob(beta_channel))
        logp += tf.reduce_sum(dist.log_prob(delta))
        logp += tf.reduce_sum(dist.log_prob(eta_channel))
        logp += tfd.LogNormal(loc=0.0, scale=1.0).log_prob(sigma)
        return logp

    dims = {"mode": "beta"}
    param_spec = [
        ParamSpec("beta0", (), tfb.Identity(), tf.constant(0.0, dtype=tf.float64)),
        ParamSpec("beta_channel", (1,), tfb.Identity(), tf.zeros([1], dtype=tf.float64)),
        ParamSpec("delta", (1,), tfb.Identity(), tf.fill([1], tf.constant(0.5, dtype=tf.float64))),
        ParamSpec("sigma", (), tfb.Softplus(), tf.constant(0.1, dtype=tf.float64)),
        ParamSpec("eta_channel", (1,), tfb.Identity(), tf.zeros([1], dtype=tf.float64)),
    ]

    samples = run_nuts(
        target_log_prob_fn,
        dims,
        param_spec,
        num_chains=1,
        num_burnin=5,
        num_samples=5,
        init_step_size=0.25,
        seed=0,
    )

    assert isinstance(samples, PosteriorSamples)
    assert samples.beta0.shape == (5, 1)
    assert samples.delta.shape[-1] == 1
