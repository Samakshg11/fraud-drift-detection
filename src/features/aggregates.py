"""REFERENCE IMPLEMENTATION for Role 3 (Feature Engineering) -- safe to replace.

Past-only, vectorised aggregates using ``groupby + rolling(closed="left")``.
The current row is always excluded. No Python row loops, no future rows.

Note: when two rows share a timestamp they are ordered by their original row
position (a nanosecond offset makes the time index unique). Only the current
row itself is excluded from its own features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_WINDOWS = ("24h", "7D")
_DEFAULT_STATS = ("mean", "count")


def past_rolling_aggregates(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = _DEFAULT_WINDOWS,
    stats: tuple[str, ...] = _DEFAULT_STATS,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Return past-only rolling aggregates as new columns (index aligned).

    Parameters
    ----------
    frame:
        Must be sorted by ``time_col`` (ascending).
    group_col:
        Entity to aggregate within, e.g. card number or customer ID.
    value_col:
        Numeric column to aggregate, e.g. transaction amount.
    windows:
        Pandas offset strings, e.g. ``("24h", "7d")``.
    stats:
        Any rolling reducer such as ``mean``, ``count``, ``sum``, ``max``.

    Returns
    -------
    DataFrame
        One column per (window, stat) pair, with the original index.
    """
    for col in (group_col, value_col, time_col):
        if col not in frame.columns:
            raise KeyError(f"Column {col!r} not found")
    if not pd.api.types.is_datetime64_any_dtype(frame[time_col]):
        raise TypeError(f"{time_col!r} must be datetime64")
    if not frame[time_col].is_monotonic_increasing:
        raise ValueError("frame must be sorted by time_col before computing aggregates")

    n = len(frame)
    if n == 0:
        return pd.DataFrame(index=frame.index)

    times = pd.DatetimeIndex(frame[time_col])
    # Unique, strictly increasing index allows safe reindexing after groupby.
    unique_index = times + pd.to_timedelta(np.arange(n), unit="ns")
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    series = pd.Series(values, index=unique_index)
    grouped = series.groupby(frame[group_col].to_numpy(), sort=False)

    output = pd.DataFrame(index=frame.index)
    name_root = prefix or f"{group_col}_{value_col}"
    for window in windows:
        rolling = grouped.rolling(window, closed="left")
        for stat in stats:
            if not hasattr(rolling, stat):
                raise ValueError(f"Unsupported rolling statistic: {stat!r}")
            aggregated = getattr(rolling, stat)()
            aligned = aggregated.droplevel(0).reindex(unique_index)
            if stat == "count":
                aligned = aligned.fillna(0)
            output[f"{name_root}_{stat}_{window}"] = pd.Series(
                aligned.to_numpy(), index=frame.index
            )
    return output


def add_velocity_features(
    frame: pd.DataFrame,
    *,
    group_col: str,
    time_col: str = "timestamp",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Seconds since the previous transaction of the same entity (past-only)."""
    if not frame[time_col].is_monotonic_increasing:
        raise ValueError("frame must be sorted by time_col")
    delta = frame.groupby(group_col, sort=False)[time_col].diff()
    column = out_col or f"{group_col}_seconds_since_previous"
    return pd.DataFrame({column: delta.dt.total_seconds()}, index=frame.index)
