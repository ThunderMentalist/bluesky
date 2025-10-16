"""ROI/ROAS utilities for the Precision MMM package.

This module provides helper functions for converting marketing mix model
contributions into return-on-investment style metrics.  The example below shows a
minimal usage pattern with dummy data.

Example
-------
>>> import numpy as np
>>> import pandas as pd
>>> from precision.hierarchy import Hierarchy
>>> from precision.summaries import Contributions
>>> from precision.roi import compute_roas
>>> idx = pd.RangeIndex(2, name="time")
>>> tactical_df = pd.DataFrame([[4.0, 6.0], [6.0, 4.0]], index=idx, columns=["t1", "t2"])
>>> platform_df = tactical_df.copy()
>>> channel_df = tactical_df.copy()
>>> controls_df = pd.DataFrame(index=idx)
>>> intercept = pd.Series([0.0, 0.0], index=idx, name="intercept")
>>> fitted = tactical_df.sum(axis=1).rename("fitted")
>>> contrib = Contributions(
...     tactical=tactical_df,
...     platform=platform_df,
...     channel=channel_df,
...     controls=controls_df,
...     intercept=intercept,
...     fitted=fitted,
...     tactical_totals=tactical_df.sum(axis=0),
...     platform_totals=platform_df.sum(axis=0),
...     channel_totals=channel_df.sum(axis=0),
...     controls_totals=controls_df.sum(axis=0),
...     intercept_total=float(intercept.sum()),
...     fitted_total=float(fitted.sum()),
... )
>>> hier = Hierarchy(
...     channel_names=["c1", "c2"],
...     platform_names=["p1", "p2"],
...     tactical_names=["t1", "t2"],
...     M_tp=np.eye(2),
...     M_tc=np.eye(2),
...     t_to_p=np.array([0, 1], dtype=int),
...     p_to_c=np.array([0, 1], dtype=int),
... )
>>> spend = pd.DataFrame([[2.0, 3.0], [3.0, 2.0]], index=idx, columns=["t1", "t2"])
>>> result = compute_roas(contributions=contrib, spend_tactical=spend, hierarchy=hier)
>>> result.tactical.loc["t1"]
2.0

Test snippet
------------
The behaviour when spend is zero can be asserted with pytest as follows::

    import numpy as np
    import pandas as pd
    from precision.roi import compute_roas

    def test_zero_spend_is_nan(contributions, hierarchy):
        spend = pd.DataFrame(0.0, index=contributions.tactical.index, columns=contributions.tactical.columns)
        roas = compute_roas(contributions=contributions, spend_tactical=spend, hierarchy=hierarchy)
        assert np.isnan(roas.tactical).all()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .hierarchy import Hierarchy
from .summaries import ContributionDraws, Contributions

Definition = Literal["roas", "roi"]
ZeroHandling = Literal["nan", "inf", "zero"]


@dataclass
class ROIResult:
    """Container for ROI/ROAS computed from contributions and spend.

    Attributes
    ----------
    tactical : pd.Series
        Total ROAS/ROI by tactical over the selected window.
    platform : pd.Series
        Total ROAS/ROI by platform over the selected window.
    channel : pd.Series
        Total ROAS/ROI by channel over the selected window.
    tactical_ts : Optional[pd.DataFrame]
        Per-time ROAS/ROI by tactical (aligned to contributions index).
    platform_ts : Optional[pd.DataFrame]
        Per-time ROAS/ROI by platform (aligned to contributions index).
    channel_ts : Optional[pd.DataFrame]
        Per-time ROAS/ROI by channel (aligned to contributions index).
    definition : {"roas", "roi"}
        Indicates whether ROAS or ROI is represented.
    window : pd.Index
        The time index used for the totals.
    """

    tactical: pd.Series
    platform: pd.Series
    channel: pd.Series
    tactical_ts: Optional[pd.DataFrame]
    platform_ts: Optional[pd.DataFrame]
    channel_ts: Optional[pd.DataFrame]
    definition: Definition
    window: pd.Index


@dataclass
class ROIInterval:
    lower: pd.Series
    upper: pd.Series


@dataclass
class ROIIntervalTS:
    lower: pd.DataFrame
    upper: pd.DataFrame


@dataclass
class ROIUncertainty:
    tactical: ROIInterval
    platform: ROIInterval
    channel: ROIInterval
    tactical_ts: Optional[ROIIntervalTS]
    platform_ts: Optional[ROIIntervalTS]
    channel_ts: Optional[ROIIntervalTS]


def _quantile_series(values: np.ndarray, index: pd.Index, q: float, name: str) -> pd.Series:
    return pd.Series(np.quantile(values, q, axis=0), index=index, name=name)


def _quantile_frame(values: np.ndarray, index: pd.Index, columns: list[str], q: float) -> pd.DataFrame:
    return pd.DataFrame(np.quantile(values, q, axis=0), index=index, columns=columns)


def _coerce_spend_to_df(
    spend_tactical: Union[pd.DataFrame, np.ndarray],
    like: pd.DataFrame,
    name: str = "spend",
) -> pd.DataFrame:
    """Return spend as a DataFrame aligned to ``like`` (same index and columns).

    Parameters
    ----------
    spend_tactical : DataFrame or ndarray
        Tactical level spend to align.
    like : DataFrame
        The DataFrame providing target index and columns (contributions.tactical).
    name : str, default "spend"
        Name used in error messages.

    Returns
    -------
    pd.DataFrame
        Spend aligned to ``like``.

    Raises
    ------
    ValueError
        If alignment fails (e.g. mismatched shape or missing labels).
    """

    if isinstance(spend_tactical, pd.DataFrame):
        aligned = spend_tactical.reindex(index=like.index, columns=like.columns)
    else:
        arr = np.asarray(spend_tactical, dtype=float)
        if arr.shape != like.shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match contributions.tactical shape {like.shape}."
            )
        aligned = pd.DataFrame(arr, index=like.index, columns=like.columns)

    if aligned.isna().any().any():
        raise ValueError(f"{name} cannot be aligned to contributions.tactical (NaNs after reindex).")
    return aligned


def _safe_divide(
    numer: Union[pd.Series, pd.DataFrame],
    denom: Union[pd.Series, pd.DataFrame],
    on_zero: ZeroHandling,
) -> Union[pd.Series, pd.DataFrame]:
    """Elementwise division with configurable zero-handling behaviour."""

    denom_arr = np.asarray(denom, dtype=float)
    numer_arr = np.asarray(numer, dtype=float)

    if denom_arr.shape != numer_arr.shape:
        raise ValueError("numerator and denominator must share the same shape")

    zero_mask = denom_arr == 0
    denom_safe = denom_arr.astype(float, copy=True)
    denom_safe[zero_mask] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer_arr / denom_safe

    if on_zero == "nan":
        pass
    elif on_zero == "inf":
        out[zero_mask] = np.inf
    elif on_zero == "zero":
        out[zero_mask] = 0.0
    else:
        raise ValueError("on_zero_spend must be 'nan', 'inf', or 'zero'.")

    if isinstance(numer, pd.DataFrame):
        return pd.DataFrame(out, index=numer.index, columns=numer.columns)
    elif isinstance(numer, pd.Series):
        return pd.Series(out, index=numer.index, name=numer.name)
    else:
        raise TypeError("numer must be a pandas Series or DataFrame")


def _safe_divide_arr(
    numer: np.ndarray,
    denom: np.ndarray,
    on_zero: ZeroHandling,
) -> np.ndarray:
    """Elementwise division for numpy arrays with configurable zero handling."""

    if numer.shape != denom.shape:
        raise ValueError("numer and denom must share the same shape")

    denom_safe = denom.astype(float, copy=True)
    zero_mask = denom_safe == 0
    denom_safe[zero_mask] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom_safe

    if on_zero == "nan":
        pass
    elif on_zero == "inf":
        out[zero_mask] = np.inf
    elif on_zero == "zero":
        out[zero_mask] = 0.0
    else:
        raise ValueError("on_zero must be 'nan', 'inf', or 'zero'.")

    return out


def compute_roi(
    *,
    contributions: Contributions,
    spend_tactical: Union[pd.DataFrame, np.ndarray],
    hierarchy: Hierarchy,
    definition: Definition = "roas",
    on_zero_spend: ZeroHandling = "nan",
    time_index: Optional[Union[pd.Index, slice, np.ndarray, list]] = None,
    return_time_series: bool = True,
) -> ROIResult:
    """Compute ROAS or ROI from contributions and tactical-level spend.

    Parameters
    ----------
    contributions : Contributions
        Outputs from ``compute_contributions_from_params`` or the ensemble.
    spend_tactical : DataFrame or ndarray
        Tactical-level spend matrix with shape ``[T, N_t]``.  If a DataFrame is
        provided it must share the same index and columns as
        ``contributions.tactical``.  An ``ndarray`` is coerced to a DataFrame
        with that same index/column layout.
    hierarchy : Hierarchy
        Provides ``M_tp`` and ``M_tc`` matrices for aggregating to platforms and
        channels.
    definition : {"roas", "roi"}, default "roas"
        ``"roas"`` computes ``contribution / spend`` while ``"roi"`` computes
        ``(contribution - spend) / spend``.
    on_zero_spend : {"nan", "inf", "zero"}, default "nan"
        Controls the output when spend is zero for a given tactical/platform/
        channel (per-period and in totals).
    time_index : slice or indexer, optional
        Restrict the total calculation to a subset of the timeline by passing a
        value understood by ``DataFrame.loc`` (e.g., a slice of dates or an
        index of periods).  The per-time series always remain aligned to the
        full contributions index.
    return_time_series : bool, default True
        If ``False``, the per-period ROAS/ROI DataFrames in the result are set
        to ``None`` to avoid unnecessary computation.

    Returns
    -------
    ROIResult
        Dataclass containing totals and per-period ROAS/ROI values.
    """

    tactical_contrib = contributions.tactical
    spend_df = _coerce_spend_to_df(spend_tactical, like=tactical_contrib, name="spend_tactical")

    if definition not in ("roas", "roi"):
        raise ValueError("definition must be either 'roas' or 'roi'.")

    numer_ts = tactical_contrib if definition == "roas" else (tactical_contrib - spend_df)

    tactical_ts = (
        _safe_divide(numer_ts, spend_df, on_zero_spend)
        if return_time_series
        else None
    )

    spend_platform = pd.DataFrame(
        spend_df.values @ hierarchy.M_tp,
        index=spend_df.index,
        columns=contributions.platform.columns,
    )
    spend_channel = pd.DataFrame(
        spend_df.values @ hierarchy.M_tc,
        index=spend_df.index,
        columns=contributions.channel.columns,
    )

    platform_contrib = contributions.platform
    channel_contrib = contributions.channel

    platform_numer = platform_contrib if definition == "roas" else (platform_contrib - spend_platform)
    channel_numer = channel_contrib if definition == "roas" else (channel_contrib - spend_channel)

    platform_ts = (
        _safe_divide(platform_numer, spend_platform, on_zero_spend)
        if return_time_series
        else None
    )
    channel_ts = (
        _safe_divide(channel_numer, spend_channel, on_zero_spend)
        if return_time_series
        else None
    )

    if time_index is None:
        window_slice = slice(None)
    else:
        window_slice = time_index

    tactical_window = numer_ts.loc[window_slice]
    platform_window = platform_numer.loc[window_slice]
    channel_window = channel_numer.loc[window_slice]

    spend_tactical_window = spend_df.loc[window_slice]
    spend_platform_window = spend_platform.loc[window_slice]
    spend_channel_window = spend_channel.loc[window_slice]

    window_index = tactical_window.index

    tactical_total = _safe_divide(tactical_window.sum(axis=0), spend_tactical_window.sum(axis=0), on_zero_spend)
    platform_total = _safe_divide(platform_window.sum(axis=0), spend_platform_window.sum(axis=0), on_zero_spend)
    channel_total = _safe_divide(channel_window.sum(axis=0), spend_channel_window.sum(axis=0), on_zero_spend)

    return ROIResult(
        tactical=tactical_total,
        platform=platform_total,
        channel=channel_total,
        tactical_ts=tactical_ts,
        platform_ts=platform_ts,
        channel_ts=channel_ts,
        definition=definition,
        window=window_index,
    )


def compute_roi_from_draws(
    *,
    contribution_draws: ContributionDraws,
    spend_tactical: Union[pd.DataFrame, np.ndarray],
    hierarchy: Hierarchy,
    definition: Definition = "roas",
    on_zero_spend: ZeroHandling = "nan",
    time_index: Optional[Union[pd.Index, slice, np.ndarray, list]] = None,
    return_time_series: bool = True,
    ci: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[ROIResult, ROIUncertainty]:
    """Compute ROI/ROAS uncertainty from per-draw contributions."""

    if definition not in ("roas", "roi"):
        raise ValueError("definition must be either 'roas' or 'roi'.")

    D, T, N_t = contribution_draws.tactical.shape
    tac_names = list(hierarchy.tactical_names)
    if len(tac_names) != N_t:
        tac_names = [f"tactical_{idx}" for idx in range(N_t)]

    if isinstance(spend_tactical, pd.DataFrame):
        if spend_tactical.shape[1] != N_t:
            raise ValueError(
                f"spend_tactical has {spend_tactical.shape[1]} columns, expected {N_t} to match contributions."
            )
        if len(spend_tactical.index) != T:
            raise ValueError(
                f"spend_tactical has {len(spend_tactical.index)} rows, expected {T} to match contributions."
            )
        like = spend_tactical.copy()
    else:
        index = pd.RangeIndex(T, name="time")
        like = pd.DataFrame(0.0, index=index, columns=tac_names)

    spend_df = _coerce_spend_to_df(spend_tactical, like=like, name="spend_tactical")
    spend_df = spend_df.astype(float)
    idx = spend_df.index
    tac_names = list(spend_df.columns)

    platform_names = list(hierarchy.platform_names)
    channel_names = list(hierarchy.channel_names)

    spend_platform = pd.DataFrame(
        spend_df.values @ hierarchy.M_tp,
        index=idx,
        columns=platform_names,
    )
    spend_channel = pd.DataFrame(
        spend_df.values @ hierarchy.M_tc,
        index=idx,
        columns=channel_names,
    )

    tac_num = (
        contribution_draws.tactical
        if definition == "roas"
        else contribution_draws.tactical - spend_df.values[None, ...]
    )
    plat_num = (
        contribution_draws.platform
        if definition == "roas"
        else contribution_draws.platform - spend_platform.values[None, ...]
    )
    chan_num = (
        contribution_draws.channel
        if definition == "roas"
        else contribution_draws.channel - spend_channel.values[None, ...]
    )

    tac_den = np.broadcast_to(spend_df.values[None, ...], (D, T, N_t))
    plat_den = np.broadcast_to(spend_platform.values[None, ...], (D, T, hierarchy.num_platforms))
    chan_den = np.broadcast_to(spend_channel.values[None, ...], (D, T, hierarchy.num_channels))

    if return_time_series:
        tac_ts = _safe_divide_arr(tac_num, tac_den, on_zero_spend)
        plat_ts = _safe_divide_arr(plat_num, plat_den, on_zero_spend)
        chan_ts = _safe_divide_arr(chan_num, chan_den, on_zero_spend)
    else:
        tac_ts = plat_ts = chan_ts = None

    if time_index is None:
        window_df = spend_df
    else:
        window_df = spend_df.loc[time_index]
    window_index = window_df.index
    window_pos = spend_df.index.get_indexer(window_index)
    if np.any(window_pos < 0):
        raise KeyError("time_index selection could not be aligned with spend index.")

    tac_tot = _safe_divide_arr(
        tac_num[:, window_pos, :].sum(axis=1),
        tac_den[:, window_pos, :].sum(axis=1),
        on_zero_spend,
    )
    plat_tot = _safe_divide_arr(
        plat_num[:, window_pos, :].sum(axis=1),
        plat_den[:, window_pos, :].sum(axis=1),
        on_zero_spend,
    )
    chan_tot = _safe_divide_arr(
        chan_num[:, window_pos, :].sum(axis=1),
        chan_den[:, window_pos, :].sum(axis=1),
        on_zero_spend,
    )

    tactical_series = pd.Series(tac_tot.mean(axis=0), index=tac_names, name=definition)
    platform_series = pd.Series(plat_tot.mean(axis=0), index=platform_names, name=definition)
    channel_series = pd.Series(chan_tot.mean(axis=0), index=channel_names, name=definition)

    if return_time_series:
        tactical_ts_mean = pd.DataFrame(tac_ts.mean(axis=0), index=idx, columns=tac_names)
        platform_ts_mean = pd.DataFrame(plat_ts.mean(axis=0), index=idx, columns=platform_names)
        channel_ts_mean = pd.DataFrame(chan_ts.mean(axis=0), index=idx, columns=channel_names)
    else:
        tactical_ts_mean = platform_ts_mean = channel_ts_mean = None

    roi_result = ROIResult(
        tactical=tactical_series,
        platform=platform_series,
        channel=channel_series,
        tactical_ts=tactical_ts_mean,
        platform_ts=platform_ts_mean,
        channel_ts=channel_ts_mean,
        definition=definition,
        window=window_index,
    )

    lower, upper = ci
    roi_uncertainty = ROIUncertainty(
        tactical=ROIInterval(
            lower=_quantile_series(tac_tot, tactical_series.index, lower, "tactical"),
            upper=_quantile_series(tac_tot, tactical_series.index, upper, "tactical"),
        ),
        platform=ROIInterval(
            lower=_quantile_series(plat_tot, platform_series.index, lower, "platform"),
            upper=_quantile_series(plat_tot, platform_series.index, upper, "platform"),
        ),
        channel=ROIInterval(
            lower=_quantile_series(chan_tot, channel_series.index, lower, "channel"),
            upper=_quantile_series(chan_tot, channel_series.index, upper, "channel"),
        ),
        tactical_ts=(
            ROIIntervalTS(
                lower=_quantile_frame(tac_ts, idx, tac_names, lower),
                upper=_quantile_frame(tac_ts, idx, tac_names, upper),
            )
            if return_time_series
            else None
        ),
        platform_ts=(
            ROIIntervalTS(
                lower=_quantile_frame(plat_ts, idx, platform_names, lower),
                upper=_quantile_frame(plat_ts, idx, platform_names, upper),
            )
            if return_time_series
            else None
        ),
        channel_ts=(
            ROIIntervalTS(
                lower=_quantile_frame(chan_ts, idx, channel_names, lower),
                upper=_quantile_frame(chan_ts, idx, channel_names, upper),
            )
            if return_time_series
            else None
        ),
    )

    return roi_result, roi_uncertainty


def compute_roas(
    *,
    contributions: Contributions,
    spend_tactical: Union[pd.DataFrame, np.ndarray],
    hierarchy: Hierarchy,
    on_zero_spend: ZeroHandling = "nan",
    time_index: Optional[Union[pd.Index, slice, np.ndarray, list]] = None,
    return_time_series: bool = True,
) -> ROIResult:
    """Compute ROAS via :func:`compute_roi` with ``definition="roas"``."""

    return compute_roi(
        contributions=contributions,
        spend_tactical=spend_tactical,
        hierarchy=hierarchy,
        definition="roas",
        on_zero_spend=on_zero_spend,
        time_index=time_index,
        return_time_series=return_time_series,
    )


__all__ = [
    "ROIResult",
    "ROIInterval",
    "ROIIntervalTS",
    "ROIUncertainty",
    "compute_roi",
    "compute_roi_from_draws",
    "compute_roas",
]
