import numpy as np
import pandas as pd
import pytest

from precision.precision.ensemble import (
    EnsembleResult,
    PerModelResult,
    _aggregate_contributions,
    _aggregate_uncertainty,
    _apply_metric_lags,
    _arrays_to_dataframe,
    _arrays_to_series,
    _check_metric_overlap,
    _compute_weights,
    _compute_weights_psis_loo,
    _fill_missing_metrics_per_tactical,
    _get_control_names,
    _make_contributions_from_arrays,
    _shift_with_lag,
    _stacking_weights_from_loglik,
    _weighted_interval_frames,
    _weighted_interval_series,
    _weighted_sum_frames,
    _weighted_sum_series,
    ensemble,
)
from precision.precision.hierarchy import Hierarchy
from precision.precision.priors import Priors
from precision.precision.summaries import (
    ContributionInterval,
    ContributionIntervalSeries,
    ContributionUncertainty,
    Contributions,
)


def _hierarchy():
    return Hierarchy(
        channel_names=["c1"],
        platform_names=["p1"],
        tactical_names=["t1", "t2"],
        M_tp=np.array([[1.0], [1.0]]),
        M_tc=np.array([[1.0], [1.0]]),
        t_to_p=np.array([0, 0], dtype=int),
        p_to_c=np.array([0], dtype=int),
    )


def _simple_contributions():
    idx = pd.RangeIndex(2, name="time")
    tactical = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=["t1", "t2"])
    platform = pd.DataFrame([[3.0], [7.0]], index=idx, columns=["p1"])
    channel = platform.copy()
    controls = pd.DataFrame(0.0, index=idx, columns=["control_0"])
    intercept = pd.Series([0.5, 0.5], index=idx, name="intercept")
    fitted = pd.Series([3.5, 7.5], index=idx, name="fitted")
    return Contributions(
        tactical=tactical,
        platform=platform,
        channel=channel,
        controls=controls,
        intercept=intercept,
        fitted=fitted,
        tactical_totals=tactical.sum(axis=0),
        platform_totals=platform.sum(axis=0),
        channel_totals=channel.sum(axis=0),
        controls_totals=controls.sum(axis=0),
        intercept_total=float(intercept.sum()),
        fitted_total=float(fitted.sum()),
    )


def _simple_uncertainty():
    idx = pd.RangeIndex(2, name="time")
    tactical = ContributionInterval(
        lower=pd.DataFrame(0.0, index=idx, columns=["t1", "t2"]),
        upper=pd.DataFrame(1.0, index=idx, columns=["t1", "t2"]),
    )
    platform = ContributionInterval(
        lower=pd.DataFrame(0.0, index=idx, columns=["p1"]),
        upper=pd.DataFrame(1.0, index=idx, columns=["p1"]),
    )
    channel = platform
    controls = ContributionInterval(
        lower=pd.DataFrame(0.0, index=idx, columns=["control_0"]),
        upper=pd.DataFrame(1.0, index=idx, columns=["control_0"]),
    )
    intercept = ContributionIntervalSeries(
        lower=pd.Series(0.0, index=idx, name="intercept"),
        upper=pd.Series(1.0, index=idx, name="intercept"),
    )
    fitted = ContributionIntervalSeries(
        lower=pd.Series(0.0, index=idx, name="fitted"),
        upper=pd.Series(1.0, index=idx, name="fitted"),
    )
    return ContributionUncertainty(
        tactical=tactical,
        platform=platform,
        channel=channel,
        controls=controls,
        intercept=intercept,
        fitted=fitted,
    )


def test_shift_with_lag_and_apply_metric_lags():
    U = np.arange(6).reshape(3, 2)
    shifted = _shift_with_lag(U, lag=1)
    assert shifted[0].sum() == 0

    metrics = {"m": U}
    lagged = _apply_metric_lags(metrics, {"m": 1})
    np.testing.assert_array_equal(lagged["m"], shifted)


def test_check_metric_overlap_detects_corr():
    y = np.arange(5, dtype=float)
    U = np.tile(y[:, None], (1, 1))
    with pytest.raises(ValueError):
        _check_metric_overlap(y, U, "metric", corr_thresh=0.5)


def test_fill_missing_metrics():
    hierarchy = _hierarchy()
    U_metrics = {"impressions": np.ones((2, hierarchy.num_tacticals)), "clicks": None, "conversions": None}
    availability = {"impressions": np.array([1, 0], dtype=np.uint8)}
    filled = _fill_missing_metrics_per_tactical(U_metrics, availability, ["impressions", "clicks", "conversions"])
    assert set(filled.keys()) == {"impressions", "clicks", "conversions"}
    assert np.all(filled["impressions"][:, 1] == 0.0)


def test_get_control_names_validation():
    assert _get_control_names(["c1"], 1) == ["c1"]
    with pytest.raises(ValueError):
        _get_control_names(["c1"], 2)


def test_arrays_to_dataframe_and_series():
    index = pd.RangeIndex(2)
    df = _arrays_to_dataframe(np.array([[1.0], [2.0]]), index, ["a"])
    assert list(df.columns) == ["a"]
    series = _arrays_to_series(np.array([1.0, 2.0]), index, "s")
    assert series.name == "s"


def test_make_contributions_from_arrays():
    hierarchy = _hierarchy()
    contrib = _make_contributions_from_arrays(
        hierarchy=hierarchy,
        control_names=["control_0"],
        tactical=np.ones((2, 2)),
        platform=np.ones((2, 1)),
        channel=np.ones((2, 1)),
        controls=np.ones((2, 1)),
        intercept=np.ones(2),
        fitted=np.ones(2),
    )
    assert isinstance(contrib, Contributions)
    assert contrib.tactical.shape == (2, 2)


def test_stacking_weights_from_loglik():
    draws = {"a": np.zeros((5, 2)), "b": np.full((5, 2), -1.0)}
    weights = _stacking_weights_from_loglik(draws, max_iter=10)
    assert weights["a"] > weights["b"]


def test_compute_weights_schemes():
    contrib = _simple_contributions()
    unc = _simple_uncertainty()
    per_model = {
        "m1": PerModelResult(
            metric="m1",
            post_mean={},
            decay_df=pd.DataFrame(),
            contributions=contrib,
            uncertainty=unc,
            log_likelihood_draws=np.zeros((2, 2)),
            r2=0.9,
            rmse=1.0,
            mae=1.0,
            sigma_mean=1.0,
            weight=np.nan,
        ),
        "m2": PerModelResult(
            metric="m2",
            post_mean={},
            decay_df=pd.DataFrame(),
            contributions=contrib,
            uncertainty=unc,
            log_likelihood_draws=np.zeros((2, 2)),
            r2=0.1,
            rmse=2.0,
            mae=2.0,
            sigma_mean=2.0,
            weight=np.nan,
        ),
    }
    weights = _compute_weights(per_model, scheme="r2", power=1.0)
    assert weights["m1"] > weights["m2"]


def test_weighted_sum_helpers():
    frames = {"a": pd.DataFrame([[1.0]], columns=["x"]), "b": pd.DataFrame([[3.0]], columns=["x"])}
    weights = {"a": 0.25, "b": 0.75}
    combined = _weighted_sum_frames(frames, weights)
    assert combined.iloc[0, 0] == 2.5

    series = {"a": pd.Series([1.0]), "b": pd.Series([3.0])}
    combined_series = _weighted_sum_series(series, weights)
    assert combined_series.iloc[0] == 2.5


def test_aggregate_contributions_and_uncertainty():
    contrib = _simple_contributions()
    unc = _simple_uncertainty()
    weights = {"m": 1.0}
    per_model = {"m": PerModelResult(
        metric="m",
        post_mean={},
        decay_df=pd.DataFrame(),
        contributions=contrib,
        uncertainty=unc,
        log_likelihood_draws=np.zeros((2, 2)),
        r2=0.0,
        rmse=0.0,
        mae=0.0,
        sigma_mean=1.0,
        weight=1.0,
    )}
    agg = _aggregate_contributions(per_model, weights)
    assert agg.tactical.equals(contrib.tactical)
    agg_unc = _aggregate_uncertainty(per_model, weights)
    assert agg_unc.tactical.lower.equals(unc.tactical.lower)


def test_weighted_intervals():
    idx = pd.RangeIndex(1)
    interval = ContributionInterval(
        lower=pd.DataFrame([[1.0]], index=idx, columns=["x"]),
        upper=pd.DataFrame([[2.0]], index=idx, columns=["x"]),
    )
    combined = _weighted_interval_frames({"a": interval}, {"a": 1.0})
    assert combined.lower.iloc[0, 0] == 1.0

    interval_series = ContributionIntervalSeries(
        lower=pd.Series([1.0], index=idx, name="s"),
        upper=pd.Series([2.0], index=idx, name="s"),
    )
    combined_series = _weighted_interval_series({"a": interval_series}, {"a": 1.0})
    assert combined_series.lower.iloc[0] == 1.0


def test_compute_weights_psis_loo(monkeypatch):
    per_model = {
        "m": PerModelResult(
            metric="m",
            post_mean={},
            decay_df=pd.DataFrame(),
            contributions=_simple_contributions(),
            uncertainty=_simple_uncertainty(),
            log_likelihood_draws=np.zeros((2, 2)),
            r2=0.0,
            rmse=0.0,
            mae=0.0,
            sigma_mean=1.0,
            weight=np.nan,
        )
    }

    def fake_psis(draws):
        return np.zeros(draws.shape[1]), np.zeros(draws.shape[1])

    def fake_stack(logdens):
        return {"m": 1.0}

    monkeypatch.setattr("precision.precision.ensemble.psis_loo_pointwise", fake_psis)
    monkeypatch.setattr(
        "precision.precision.ensemble.stacking_weights_from_pointwise_logdens",
        fake_stack,
    )
    weights = _compute_weights_psis_loo(per_model)
    assert weights["m"] == 1.0


def test_ensemble_high_level(monkeypatch):
    hierarchy = _hierarchy()
    y = np.array([1.0, 2.0])
    U_metrics = {"impressions": np.ones((2, 2)), "clicks": np.ones((2, 2)), "conversions": np.ones((2, 2))}

    dummy_result = PerModelResult(
        metric="impressions",
        post_mean={},
        decay_df=pd.DataFrame(),
        contributions=_simple_contributions(),
        uncertainty=_simple_uncertainty(),
        log_likelihood_draws=np.zeros((2, 2)),
        r2=0.5,
        rmse=1.0,
        mae=1.0,
        sigma_mean=1.0,
        weight=1.0,
    )

    def fake_fit(**kwargs):  # type: ignore[no-untyped-def]
        return dummy_result

    monkeypatch.setattr(
        "precision.precision.ensemble._fit_single_metric_model", lambda **kwargs: dummy_result
    )
    monkeypatch.setattr(
        "precision.precision.ensemble._compute_weights",
        lambda per_model, scheme, power: {k: 1.0 for k in per_model},
    )

    result = ensemble(
        hierarchy=hierarchy,
        y=y,
        U_metrics=U_metrics,
        priors=Priors(),
        nuts_args={"num_chains": 1, "num_burnin": 1, "num_samples": 1},
    )

    assert isinstance(result, EnsembleResult)
    assert set(result.weights.keys()) == {"impressions", "clicks", "conversions"}
