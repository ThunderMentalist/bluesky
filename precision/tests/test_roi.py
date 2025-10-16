import numpy as np
import pandas as pd
import pytest

from precision.precision.hierarchy import Hierarchy
from precision.precision.roi import (
    ROIInterval,
    ROIIntervalTS,
    ROIResult,
    ROIUncertainty,
    _coerce_spend_to_df,
    _quantile_frame,
    _quantile_series,
    _safe_divide,
    _safe_divide_arr,
    compute_roi,
    compute_roi_from_draws,
    compute_roas,
)
from precision.precision.summaries import ContributionDraws, Contributions


@pytest.fixture()
def simple_hierarchy():
    return Hierarchy(
        channel_names=["c1", "c2"],
        platform_names=["p1", "p2"],
        tactical_names=["t1", "t2"],
        M_tp=np.eye(2),
        M_tc=np.eye(2),
        t_to_p=np.array([0, 1], dtype=int),
        p_to_c=np.array([0, 1], dtype=int),
    )


@pytest.fixture()
def contributions(simple_hierarchy):
    idx = pd.RangeIndex(3, name="time")
    tactical = pd.DataFrame([[1.0, 2.0], [2.0, 0.0], [1.0, 1.0]], index=idx, columns=simple_hierarchy.tactical_names)
    platform = tactical.copy()
    channel = tactical.copy()
    controls = pd.DataFrame(0.0, index=idx, columns=["c"])
    intercept = pd.Series([0.5, 0.5, 0.5], index=idx, name="intercept")
    fitted = tactical.sum(axis=1) + controls.sum(axis=1) + intercept
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


@pytest.fixture()
def spend_df(contributions):
    return contributions.tactical.copy()


def test_quantile_helpers(contributions):
    arr_series = np.arange(15, dtype=float).reshape(5, 3)
    arr_frame = np.arange(45, dtype=float).reshape(5, 3, 3)
    index = contributions.tactical.index
    columns = list("abc")
    series = _quantile_series(arr_series, index, 0.5, "median")
    frame = _quantile_frame(arr_frame, index, columns, 0.75)

    assert isinstance(series, pd.Series)
    assert series.name == "median"
    assert frame.columns.tolist() == columns
    assert frame.shape == (len(index), len(columns))


def test_coerce_spend_to_df_validates_shape(contributions):
    arr = contributions.tactical.to_numpy()
    out = _coerce_spend_to_df(arr, like=contributions.tactical)
    pd.testing.assert_frame_equal(out, contributions.tactical)

    with pytest.raises(ValueError):
        _coerce_spend_to_df(arr[:, :1], like=contributions.tactical)


def test_safe_divide_handles_zero(contributions):
    numer = contributions.tactical.iloc[0]
    denom = numer.copy()
    denom.iloc[0] = 0.0
    result = _safe_divide(numer, denom, on_zero="inf")
    assert np.isinf(result.iloc[0])

    arr_result = _safe_divide_arr(np.array([1.0, 0.0]), np.array([0.0, 2.0]), on_zero="zero")
    assert arr_result[0] == 0.0


def test_compute_roi_returns_expected(contributions, spend_df, simple_hierarchy):
    roi = compute_roi(contributions=contributions, spend_tactical=spend_df, hierarchy=simple_hierarchy)
    assert isinstance(roi, ROIResult)
    assert roi.definition == "roas"
    pd.testing.assert_series_equal(roi.tactical, (contributions.tactical.sum() / spend_df.sum()))


def test_compute_roas_alias(contributions, spend_df, simple_hierarchy):
    roas = compute_roas(contributions=contributions, spend_tactical=spend_df, hierarchy=simple_hierarchy)
    assert roas.definition == "roas"


def test_compute_roi_from_draws_uncertainty(simple_hierarchy, spend_df):
    idx = pd.RangeIndex(3, name="time")
    draws = ContributionDraws(
        tactical=np.ones((5, 3, 2)),
        platform=np.ones((5, 3, 2)),
        channel=np.ones((5, 3, 2)),
        controls=np.zeros((5, 3, 0)),
        intercept=np.ones((5, 3)),
        fitted=np.ones((5, 3)),
    )

    result, uncertainty = compute_roi_from_draws(
        contribution_draws=draws,
        spend_tactical=spend_df,
        hierarchy=simple_hierarchy,
        definition="roas",
        return_time_series=True,
    )

    assert isinstance(result, ROIResult)
    assert isinstance(uncertainty, ROIUncertainty)
    assert isinstance(uncertainty.tactical, ROIInterval)
    assert isinstance(uncertainty.tactical_ts, ROIIntervalTS)
