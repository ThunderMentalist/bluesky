import numpy as np

from precision.precision.psis import (
    _logsumexp,
    _psis_smooth_log_weights,
    psis_loo_pointwise,
    stacking_weights_from_pointwise_logdens,
)


def test_logsumexp_matches_numpy():
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert np.allclose(_logsumexp(x, axis=0), np.log(np.sum(np.exp(x), axis=0)))


def test_psis_smoothing_preserves_mass():
    rng = np.random.default_rng(0)
    log_raw = rng.normal(size=50)
    smoothed = _psis_smooth_log_weights(log_raw)
    weights = np.exp(smoothed)
    assert np.allclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_psis_loo_pointwise_reasonable():
    draws = np.array([[ -0.5, -0.1], [ -0.2, -0.3], [ -1.0, -0.4]])
    loo, k = psis_loo_pointwise(draws)
    assert loo.shape == (2,)
    assert k.shape == (2,)
    assert np.all(k >= 0.0)


def test_stacking_weights_simple_case():
    logdens = {
        "m1": np.log(np.array([0.7, 0.6, 0.8])),
        "m2": np.log(np.array([0.3, 0.4, 0.2])),
    }
    weights = stacking_weights_from_pointwise_logdens(logdens, lr=0.5, max_iter=500)
    assert set(weights.keys()) == {"m1", "m2"}
    assert weights["m1"] > weights["m2"]
    assert np.isclose(sum(weights.values()), 1.0)
