import numpy as np

from precision.precision.hierarchy import build_hierarchy
from precision.precision.summaries import (
    compute_contribution_arrays,
    log_likelihood_draws_single_model,
)
from precision.precision.priors import Priors
from precision.precision.sampling import PosteriorSamples


def _toy_hierarchy():
    tree = {"channelA": {"platformA": ["t1", "t2"]}}
    levels = ["tactical", "platform", "channel"]
    return build_hierarchy(tree, levels)


def test_compute_contribution_arrays_scaling_semantics():
    H = _toy_hierarchy()
    U = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=float)
    delta = np.array([0.0, 0.0], dtype=float)
    beta = np.array([1.0, 1.0], dtype=float)
    contrib = compute_contribution_arrays(
        U_raw=U,
        delta=delta,
        beta=beta,
        beta_level="tactical",
        hierarchy=H,
        leaf_level="tactical",
        normalize_adstock=False,
        use_saturation=False,
        s_sat=None,
        tactical_rescale=np.array([0.5, 2.0], dtype=float),
    )
    expected_leaf = U * np.array([0.5, 2.0])[None, :]
    assert np.allclose(contrib["tactical"], expected_leaf)


def test_loglik_helper_shapes_iid_normal():
    H = _toy_hierarchy()
    T, N = 5, 2
    y = np.zeros(T, dtype=float)
    U = np.zeros((T, N), dtype=float)
    Z = np.zeros((T, 0), dtype=float)
    priors = Priors(
        center_y=True,
        standardize_media="none",
        likelihood="normal",
        residual_mode="iid",
        beta_structure="tactical_hier",
    )
    draws = 3
    chains = 1
    beta0 = np.zeros((chains, draws), dtype=float)
    sigma = np.ones((chains, draws), dtype=float)
    delta = np.zeros((chains, draws, N), dtype=float)
    beta_by_level = {"tactical": np.ones((chains, draws, N), dtype=float)}
    samples = PosteriorSamples(
        beta0=beta0,
        beta_channel=np.zeros((chains, draws, 0)),
        gamma=np.zeros((chains, draws, 0)),
        delta=delta,
        sigma=sigma,
        half_life=None,
        logit_delta=None,
        mu_c=None,
        tau_c=None,
        beta_platform=None,
        beta_tactical=beta_by_level["tactical"],
        tau_beta=None,
        tau0=None,
        lambda_local=None,
        s_sat=None,
        phi=None,
        eta_channel=None,
        eta_platform=None,
        eta_tactical=None,
        eta_by_level={},
        beta_by_level=beta_by_level,
    )
    L = log_likelihood_draws_single_model(
        y=y,
        U_tactical=U,
        Z_controls=Z,
        hierarchy=H,
        priors=priors,
        samples=samples,
        normalize_adstock=True,
    )
    assert L.shape == (draws, T)
