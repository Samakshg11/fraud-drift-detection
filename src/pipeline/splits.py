"""Chronological month-based splits and walk-forward validation.

This module is the heart of Role 2. Three invariants are enforced everywhere:

1. Splits are defined by *calendar month*, never by random row assignment.
2. ``max(train_time) < min(test_time)`` is asserted for every produced split.
3. An optional embargo (``gap_days``) drops the first days of the test block,
   because fraud labels arrive late (chargebacks).

All functions work on a DataFrame that already has a datetime ``time_col`` and
an integer ``month_col`` (see :func:`add_month_index`). Indices returned by the
generator are labels of ``df.index``, so callers can use ``df.loc[idx]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd

MONTH_COLUMN = "month"
_TIME_UNITS = ("datetime", "seconds", "unix_seconds")


@dataclass(frozen=True)
class TimeSplit:
    """One chronological train/test split, with metadata for reporting."""

    train_idx: pd.Index
    test_idx: pd.Index
    train_months: tuple[int, ...]
    test_months: tuple[int, ...]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __iter__(self) -> Iterator[pd.Index]:
        """Allow ``train_idx, test_idx = split`` (KFold-style unpacking)."""
        yield self.train_idx
        yield self.test_idx


def derive_datetime(
    values: pd.Series,
    *,
    reference_date: str | pd.Timestamp | None = None,
    time_unit: str = "datetime",
) -> pd.Series:
    """Convert a raw time column to timezone-naive datetimes.

    Parameters
    ----------
    values:
        Raw time values. For IEEE-CIS these are seconds offsets from an
        unknown reference (``TransactionDT``); for Sparkov they are datetime
        strings.
    reference_date:
        Anchor used when ``time_unit="seconds"``. Required in that case.
    time_unit:
        ``"datetime"`` (parse strings/timestamps), ``"seconds"`` (offset from
        ``reference_date``), or ``"unix_seconds"`` (absolute epoch seconds).
    """
    if time_unit not in _TIME_UNITS:
        raise ValueError(f"time_unit must be one of {_TIME_UNITS}, got {time_unit!r}")

    if time_unit == "datetime":
        result = pd.to_datetime(values, errors="raise")
    elif time_unit == "unix_seconds":
        result = pd.to_datetime(pd.to_numeric(values), unit="s", errors="raise")
    else:
        if reference_date is None:
            raise ValueError("reference_date is required when time_unit='seconds'")
        offsets = pd.to_timedelta(pd.to_numeric(values), unit="s", errors="raise")
        result = pd.Timestamp(reference_date) + offsets

    if getattr(result.dtype, "tz", None) is not None:
        result = result.dt.tz_localize(None)
    result.name = values.name
    return result


def month_index_from_datetime(values: pd.Series) -> pd.Series:
    """Map datetimes to a sortable integer month index (``year*12 + month-1``).

    Using an absolute index rather than ``1..12`` makes December->January
    transitions correct and lets us add months with plain integer arithmetic.
    """
    dt = pd.to_datetime(values)
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    result = (dt.dt.year.astype("int64") * 12 + dt.dt.month.astype("int64") - 1)
    result.name = values.name
    return result


def period_from_month_index(index: int) -> pd.Period:
    """Convert an integer month index back to a ``pandas.Period`` (freq='M')."""
    if not isinstance(index, (int,)) or isinstance(index, bool):
        raise TypeError(f"month index must be an int, got {index!r}")
    return pd.Period(year=index // 12, month=index % 12 + 1, freq="M")


def month_bounds(index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(first_instant, first_instant_of_next_month)`` for a month."""
    period = period_from_month_index(index)
    return period.to_timestamp(), (period + 1).to_timestamp()


def add_month_index(
    df: pd.DataFrame,
    *,
    time_col: str,
    reference_date: str | pd.Timestamp | None = None,
    time_unit: str = "datetime",
    out_col: str = MONTH_COLUMN,
) -> pd.DataFrame:
    """Return ``df`` with an added integer month column.

    A new DataFrame is returned (via ``assign``); the input is never mutated.
    Only one column is added, so the copy is cheap even for wide frames.
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col {time_col!r} not found in DataFrame")
    times = derive_datetime(df[time_col], reference_date=reference_date, time_unit=time_unit)
    return df.assign(**{out_col: month_index_from_datetime(times)})


def _validate_time_columns(df: pd.DataFrame, time_col: str, month_col: str) -> None:
    for col in (time_col, month_col):
        if col not in df.columns:
            raise KeyError(f"Required column {col!r} not found in DataFrame")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise TypeError(
            f"{time_col!r} must be datetime64; call derive_datetime/add_month_index first"
        )


def _split_at_anchor(
    df: pd.DataFrame,
    *,
    time_col: str,
    month_col: str,
    train_start_month: int,
    anchor_month: int,
    test_months: int,
    gap_days: int,
    allow_empty_test: bool = False,
) -> TimeSplit | None:
    """Build a split from explicit month bounds. Internal helper."""
    if train_start_month > anchor_month:
        raise ValueError("train_start_month must be <= anchor_month")
    if test_months <= 0:
        raise ValueError("test_months must be positive")
    if gap_days < 0:
        raise ValueError("gap_days must be non-negative")

    months = df[month_col]
    train_mask = months.between(train_start_month, anchor_month)
    test_months_range = range(anchor_month + 1, anchor_month + test_months + 1)
    test_mask = months.isin(test_months_range)

    first_test_instant, _ = month_bounds(anchor_month + 1)
    embargo_cutoff = first_test_instant + pd.Timedelta(days=gap_days)
    test_mask &= df[time_col] >= embargo_cutoff

    train_idx = df.index[train_mask]
    test_idx = df.index[test_mask]
    if train_idx.empty:
        raise ValueError(
            f"No training rows between months {train_start_month} and {anchor_month}"
        )
    if test_idx.empty:
        if allow_empty_test:
            return None
        raise ValueError(
            f"No test rows after applying embargo of {gap_days} day(s) "
            f"to months {list(test_months_range)}"
        )

    train_end = df.loc[train_idx, time_col].max()
    test_start = df.loc[test_idx, time_col].min()
    if train_end >= test_start:
        raise AssertionError(
            f"Temporal ordering violated: max(train_time)={train_end} "
            f">= min(test_time)={test_start}"
        )
    if test_start < embargo_cutoff:
        raise AssertionError(
            f"Embargo violated: test_start={test_start} < cutoff={embargo_cutoff}"
        )

    return TimeSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        train_months=tuple(range(train_start_month, anchor_month + 1)),
        test_months=tuple(test_months_range),
        train_end=train_end,
        test_start=test_start,
        test_end=df.loc[test_idx, time_col].max(),
    )


def single_time_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    month_col: str = MONTH_COLUMN,
    train_months: int,
    test_months: int,
    gap_days: int = 0,
    anchor_month: int | None = None,
) -> TimeSplit:
    """Create one chronological split.

    By default the *most recent* complete window is used: the last
    ``test_months`` months of the data are the test block and the preceding
    ``train_months`` are training. Pass ``anchor_month`` to pin the last
    training month explicitly.
    """
    _validate_time_columns(df, time_col, month_col)
    observed = sorted(df[month_col].unique().tolist())
    if len(observed) < 2:
        raise ValueError("Need at least two distinct months to make a time split")

    if anchor_month is None:
        anchor_month = observed[-1] - test_months
    if anchor_month not in observed:
        raise ValueError(
            f"anchor_month={anchor_month} is not an observed month; "
            f"observed range is {observed[0]}..{observed[-1]}"
        )
    train_start = anchor_month - train_months + 1
    split = _split_at_anchor(
        df,
        time_col=time_col,
        month_col=month_col,
        train_start_month=train_start,
        anchor_month=anchor_month,
        test_months=test_months,
        gap_days=gap_days,
    )
    assert split is not None  # allow_empty_test=False guarantees this
    return split


def walk_forward_splits(
    df: pd.DataFrame,
    *,
    time_col: str,
    month_col: str = MONTH_COLUMN,
    train_months: int,
    test_months: int,
    gap_days: int = 0,
    scheme: str = "expanding",
    sliding_window_months: int | None = None,
) -> Iterator[TimeSplit]:
    """Yield chronological folds with non-overlapping test blocks.

    ``expanding``: training always starts at the first observed month.
    ``sliding``: training is capped at ``sliding_window_months``.

    Test blocks advance by ``test_months`` each fold, so no row is ever used
    for testing twice. Folds whose test block is fully removed by the embargo
    are skipped.
    """
    _validate_time_columns(df, time_col, month_col)
    if scheme not in ("expanding", "sliding"):
        raise ValueError(f"scheme must be 'expanding' or 'sliding', got {scheme!r}")
    if scheme == "sliding":
        if sliding_window_months is None or sliding_window_months <= 0:
            raise ValueError("sliding_window_months must be positive for sliding scheme")
        if sliding_window_months > train_months:
            raise ValueError("sliding_window_months cannot exceed train_months")

    observed = sorted(df[month_col].unique().tolist())
    first_month, last_month = observed[0], observed[-1]
    first_anchor = first_month + train_months - 1
    last_anchor = last_month - test_months

    if first_anchor > last_anchor:
        raise ValueError(
            f"Not enough months for {train_months}-month train / {test_months}-month "
            f"test windows: observed {last_month - first_month + 1} month(s)"
        )

    for anchor in range(first_anchor, last_anchor + 1, test_months):
        if scheme == "expanding":
            train_start = first_month
        else:
            train_start = max(first_month, anchor - sliding_window_months + 1)
        split = _split_at_anchor(
            df,
            time_col=time_col,
            month_col=month_col,
            train_start_month=train_start,
            anchor_month=anchor,
            test_months=test_months,
            gap_days=gap_days,
            allow_empty_test=True,
        )
        if split is not None:
            yield split


def walk_forward_splits_from_config(
    df: pd.DataFrame,
    config: dict,
    *,
    time_col: str | None = None,
    month_col: str = MONTH_COLUMN,
) -> Iterator[TimeSplit]:
    """Convenience wrapper that reads all split settings from ``config``."""
    split_cfg = config["split"]
    return walk_forward_splits(
        df,
        time_col=time_col or split_cfg["time_col"],
        month_col=month_col,
        train_months=split_cfg["train_months"],
        test_months=split_cfg["test_months"],
        gap_days=split_cfg["gap_days"],
        scheme=split_cfg["scheme"],
        sliding_window_months=split_cfg["sliding_window_months"],
    )


def describe_splits(splits: Sequence[TimeSplit]) -> pd.DataFrame:
    """Summarise folds for logging/reports (no model results, just metadata)."""
    rows = []
    for fold, split in enumerate(splits):
        rows.append(
            {
                "fold": fold,
                "train_rows": len(split.train_idx),
                "test_rows": len(split.test_idx),
                "train_start_month": split.train_months[0],
                "train_end_month": split.train_months[-1],
                "test_start_month": split.test_months[0],
                "test_end_month": split.test_months[-1],
                "train_end": split.train_end,
                "test_start": split.test_start,
            }
        )
    return pd.DataFrame(rows)
