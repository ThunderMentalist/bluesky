import importlib.util
import pytest


def test_mmm_pipeline_runs():
    required = {
        "numpy": "numpy is required for the MMM pipeline test",
        "tensorflow": "tensorflow is required for the MMM pipeline test",
        "tensorflow_probability": "tensorflow_probability is required for the MMM pipeline test",
    }

    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(
            ", ".join(required[name] for name in missing),
        )

    import numpy as np  # type: ignore[import-untyped]
    import tensorflow as tf  # noqa: F401
    import tensorflow_probability as tfp  # noqa: F401

    from precision import (
        Priors,
        build_hierarchy,
        compute_contributions_from_params,
        make_target_log_prob_fn,
        posterior_mean,
        run_nuts,
    )

    rng = np.random.default_rng(0)
    spec = {
        "channel100": {"platform110": ["tactical111", "tactical112"]},
        "channel200": {"platform210": ["tactical211", "tactical212"]},
        "channel300": {"platform310": ["tactical311", "tactical312"]},
    }
    hierarchy = build_hierarchy(spec)

    T = 12
    num_tacticals = hierarchy.num_tacticals
    U_tactical = rng.gamma(shape=2.0, scale=1.0, size=(T, num_tacticals))
    Z_controls = rng.normal(size=(T, 1))

    true_delta = rng.beta(a=2, b=5, size=num_tacticals)
    adstocked = np.zeros_like(U_tactical)
    for t in range(T):
        if t == 0:
            adstocked[t] = U_tactical[t]
        else:
            adstocked[t] = U_tactical[t] + true_delta * adstocked[t - 1]
    beta_true = np.array([0.4, 0.3, 0.2])
    beta_tactical = hierarchy.M_tc @ beta_true
    channel_signal = (adstocked * beta_tactical[None, :]) @ hierarchy.M_tc
    gamma_true = np.array([0.1])
    mean = 0.8 + channel_signal.sum(axis=1) + Z_controls[:, 0] * gamma_true[0]
    y = mean + rng.normal(scale=0.5, size=T)

    target_fn, dims, param_spec = make_target_log_prob_fn(
        y=y,
        U_tactical=U_tactical,
        Z_controls=Z_controls,
        hierarchy=hierarchy,
        priors=Priors(beta0_sd=2.0, beta_sd=1.5, gamma_sd=1.5, sigma_sd=1.0),
    )

    samples = run_nuts(
        target_fn,
        dims,
        param_spec,
        num_chains=1,
        num_burnin=10,
        num_samples=15,
        init_step_size=0.1,
        seed=1,
    )

    post = posterior_mean(samples)
    contrib = compute_contributions_from_params(
        y=y,
        U_tactical=U_tactical,
        Z_controls=Z_controls,
        control_names=["control"],
        hierarchy=hierarchy,
        beta0=post["beta0"],
        beta_channel=post["beta_channel"],
        gamma=post["gamma"],
        delta=post["delta"],
    )

    assert contrib.channel.shape == (T, hierarchy.num_channels)
    assert np.isfinite(post["beta0"]).all()
    assert np.isfinite(post["beta_channel"]).all()
    assert np.isfinite(post["delta"]).all()
