import numpy as np
import pandas as pd

from precision.precision.hierarchy import build_hierarchy
from precision.precision.sampling import PosteriorSamples
from precision.precision.summaries import (
    ContributionDraws,
    ContributionInterval,
    ContributionIntervalSeries,
    ContributionSignProb,
    Contributions,
    _saturate_log1p_np,
    beta_per_leaf_from_params,
    beta_per_tactical_from_params,
    compute_contribution_arrays,
    compute_contributions_from_params,
    contribution_sign_probabilities,
    contributions_from_posterior,
    posterior_mean,
    summarise_decay_rates,
)


def _hierarchy():
    return build_hierarchy(
        {"c1": {"p1": ["t1", "t2"]}}, ["tactical", "platform", "channel"]
    )


def test_compute_contribution_arrays_basic():
    hierarchy = _hierarchy()
    U = np.ones((3, 2))
    delta = np.array([0.3, 0.3])
    beta_channel = np.array([0.2])
    contribs = compute_contribution_arrays(
        U_raw=U,
        delta=delta,
        beta=beta_channel,
        beta_level="channel",
        hierarchy=hierarchy,
        leaf_level="tactical",
        normalize_adstock=False,
        use_saturation=False,
        s_sat=None,
    )
    assert set(contribs) == {"tactical", "platform", "channel"}
    assert contribs["tactical"].shape == (3, 2)
    assert contribs["platform"].shape == (3, 1)
    assert contribs["channel"].shape == (3, 1)


def test_contributions_from_posterior():
    hierarchy = _hierarchy()
    samples = PosteriorSamples(
        beta0=np.zeros((1, 1, 1)),
        beta_channel=np.zeros((1, 1, 1)),
        gamma=np.zeros((1, 1, 0)),
        delta=np.full((1, 1, 2), 0.5),
        sigma=np.ones((1, 1, 1)),
    )
    y = np.array([1.0, 2.0], dtype=float)
    U = np.ones((2, 2))
    Z = None
    contributions, uncertainty, draws = contributions_from_posterior(
        samples,
        y=y,
        U_tactical=U,
        Z_controls=Z,
        hierarchy=hierarchy,
        normalize_adstock=False,
        return_draws=True,
    )
    assert isinstance(contributions, Contributions)
    assert isinstance(uncertainty.tactical, ContributionInterval)
    assert isinstance(uncertainty.intercept, ContributionIntervalSeries)
    assert isinstance(draws, ContributionDraws)


def test_contribution_sign_probabilities():
    hierarchy = _hierarchy()
    draws = ContributionDraws(
        tactical=np.array([[[1.0, -1.0]], [[-1.0, 1.0]]]),
        platform=np.array([[[1.0]], [[-1.0]]]),
        channel=np.array([[[1.0]], [[-1.0]]]),
        controls=np.zeros((2, 1, 0)),
        intercept=np.zeros((2, 1)),
        fitted=np.zeros((2, 1)),
    )
    prob = contribution_sign_probabilities(draws, hierarchy=hierarchy)
    assert isinstance(prob, ContributionSignProb)
    assert prob.tactical_ts.shape == (1, 2)


def test_posterior_mean_handles_none():
    samples = PosteriorSamples(
        beta0=np.zeros((1, 1, 1)),
        beta_channel=np.zeros((1, 1, 1)),
        gamma=np.zeros((1, 1, 0)),
        delta=np.ones((1, 1, 1)),
        sigma=np.ones((1, 1, 1)),
    )
    mean = posterior_mean(samples)
    assert "beta0" in mean and mean["beta0"] == 0.0


def test_saturate_log1p_np():
    x = np.array([0.0, 1.0])
    out = _saturate_log1p_np(x, 0.5)
    assert out[1] < 1.0


def test_beta_per_tactical_from_params_prefers_tactical():
    hierarchy = _hierarchy()
    beta_leaf = beta_per_leaf_from_params(
        beta=np.array([1.5]),
        beta_level="channel",
        hierarchy=hierarchy,
        leaf_level="tactical",
    )
    np.testing.assert_array_equal(beta_leaf, np.array([1.5, 1.5]))

    beta = beta_per_tactical_from_params(hierarchy, beta_tactical=np.array([1.0, 2.0]))
    np.testing.assert_array_equal(beta, np.array([1.0, 2.0]))

    beta = beta_per_tactical_from_params(hierarchy, beta_platform=np.array([1.0]))
    np.testing.assert_array_equal(beta, np.array([1.0, 1.0]))


def test_summarise_decay_rates():
    hierarchy = _hierarchy()
    samples = PosteriorSamples(
        beta0=np.zeros((1, 1, 1)),
        beta_channel=np.zeros((1, 1, 1)),
        gamma=np.zeros((1, 1, 0)),
        delta=np.ones((1, 1, 2)) * 0.5,
        sigma=np.ones((1, 1, 1)),
    )
    df = summarise_decay_rates(samples, hierarchy)
    assert set(df.columns) >= {"tactical", "decay_mean"}


def test_compute_contributions_from_params():
    hierarchy = _hierarchy()
    y = np.array([1.0, 2.0])
    U = np.ones((2, 2))
    contrib = compute_contributions_from_params(
        y,
        U,
        Z_controls=None,
        control_names=None,
        hierarchy=hierarchy,
        beta0=np.array([0.0]),
        beta_channel=np.array([0.5]),
        gamma=np.zeros((0,)),
        delta=np.array([0.3, 0.3]),
        normalize_adstock=False,
    )
    assert isinstance(contrib, Contributions)
    assert contrib.tactical.shape == (2, 2)
