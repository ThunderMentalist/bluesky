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
    try:
        import tensorflow_probability as tfp  # noqa: F401  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"tensorflow_probability unavailable: {exc}")

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
        beta_platform=post.get("beta_platform"),
        beta_tactical=post.get("beta_tactical"),
    )

    assert contrib.channel.shape == (T, hierarchy.num_channels)
    assert np.isfinite(post["beta0"]).all()
    assert np.isfinite(post["beta_channel"]).all()
    assert np.isfinite(post["delta"]).all()


def test_beta_per_tactical_helper_priority():
    import numpy as np  # type: ignore[import-untyped]

    try:
        from precision import build_hierarchy
        from precision.precision.summaries import beta_per_tactical_from_params
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"precision package unavailable due to dependency error: {exc}")

    hierarchy = build_hierarchy({"c1": {"p1": ["t1", "t2"], "p2": ["t3"]}})

    beta_channel = np.array([1.0], dtype=float)
    expected_channel = hierarchy.M_tc @ beta_channel
    np.testing.assert_allclose(
        beta_per_tactical_from_params(hierarchy, beta_channel=beta_channel),
        expected_channel,
    )

    beta_platform = np.array([0.5, 0.8], dtype=float)
    expected_platform = hierarchy.M_tp @ beta_platform
    np.testing.assert_allclose(
        beta_per_tactical_from_params(
            hierarchy, beta_channel=beta_channel, beta_platform=beta_platform
        ),
        expected_platform,
    )

    beta_tactical = np.array([0.1, 0.2, 0.3], dtype=float)
    np.testing.assert_allclose(
        beta_per_tactical_from_params(
            hierarchy,
            beta_channel=beta_channel,
            beta_platform=beta_platform,
            beta_tactical=beta_tactical,
        ),
        beta_tactical,
    )


@pytest.mark.parametrize(
    "beta_structure,sparsity",
    [
        ("channel", "none"),
        ("platform_hier", "none"),
        ("tactical_hier", "none"),
        ("tactical_hier", "horseshoe"),
    ],
)
def test_make_target_log_prob_fn_beta_structures_smoke(beta_structure: str, sparsity: str):
    required = {
        "numpy": "numpy is required for the beta structure smoke test",
        "tensorflow": "tensorflow is required for the beta structure smoke test",
        "tensorflow_probability": "tensorflow_probability is required for the beta structure smoke test",
    }

    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(
            ", ".join(required[name] for name in missing),
        )

    import numpy as np  # type: ignore[import-untyped]
    import tensorflow as tf  # type: ignore[import-untyped]

    try:
        from precision import Priors, build_hierarchy, make_target_log_prob_fn
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"precision package unavailable due to dependency error: {exc}")

    spec = {"c1": {"p1": ["t1", "t2"], "p2": ["t3"]}}
    hierarchy = build_hierarchy(spec)
    T = 5
    N_t = hierarchy.num_tacticals
    y = np.zeros(T, dtype=float)
    U = np.ones((T, N_t), dtype=float)

    priors = Priors(beta_structure=beta_structure, sparsity_prior=sparsity)
    target_fn, _dims, param_spec = make_target_log_prob_fn(
        y=y,
        U_tactical=U,
        Z_controls=None,
        hierarchy=hierarchy,
        priors=priors,
    )

    params = [tf.convert_to_tensor(p.init) for p in param_spec]
    value = target_fn(*params)
    assert value.shape == ()
    assert tf.math.is_finite(value)
