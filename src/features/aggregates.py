"""Past-only rolling aggregate features for fraud detection (Role 3 — Harsh).

Design principles
-----------------
* **No leakage**: every aggregate uses ``closed="left"`` so the current row is
  *never* included in its own feature.  Rows sharing a timestamp are ordered by
  their original position (a nanosecond offset on the index makes the DatetimeIndex
  strictly increasing).
* **Vectorised**: no Python-level row loops.  ``groupby + rolling`` operates on
  NumPy arrays internally.
* **Multi-entity**: card-level, merchant-level, and card×merchant combo aggregates
  capture different fraud signals.

Extended relative to the reference skeleton
-------------------------------------------
* Additional statistics: ``std``, ``max``, ``sum`` (reference had only ``mean``
  and ``count``).
* Merchant-level aggregates (reference had card-level only).
* Card × merchant interaction aggregates — a known high-signal feature for
  detecting fraud at a specific terminal.
* Amount-deviation feature: how far the current amount is from the card's rolling
  mean (standardised, past-only).
* High-spend spike detector: flag transactions whose amount exceeds the card's
  rolling 7-day ``mean + 2*std``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_WINDOWS = ("1h", "24h", "7D")
_EXTENDED_STATS = ("mean", "count", "sum", "std", "max")


def _make_unique_time_index(times: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Add nanosecond offsets so that every index position is strictly unique.

    This is required so that ``droplevel(0).reindex(unique_index)`` maps each
    aggregate value back to exactly one row even when multiple transactions share
    the same timestamp.
    """
    return times + pd.to_timedelta(np.arange(n), unit="ns")


def _validate_frame(
    frame: pd.DataFrame, group_col: str, value_col: str, time_col: str
) -> None:
    for col in (group_col, value_col, time_col):
        if col not in frame.columns:
            raise KeyError(f"Column {col!r} not found in frame")
    if not pd.api.types.is_datetime64_any_dtype(frame[time_col]):
        raise TypeError(f"{time_col!r} must be a datetime64 column")
    if not frame[time_col].is_monotonic_increasing:
        raise ValueError(
            f"frame must be sorted by '{time_col}' (ascending) before computing "
            "aggregates"
        )


# ---------------------------------------------------------------------------
# Core rolling-aggregate builder
# ---------------------------------------------------------------------------


def past_rolling_aggregates(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = ("24h", "7D"),
    stats: tuple[str, ...] = ("mean", "count"),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Return past-only rolling aggregates as new columns (index aligned).

    Parameters
    ----------
    frame:
        Must be sorted ascending by ``time_col``.
    group_col:
        Entity column to group within (e.g. card ID, merchant ID).
    value_col:
        Numeric column to aggregate (e.g. transaction amount).
    windows:
        Pandas offset strings such as ``("1h", "24h", "7D")``.
    stats:
        Rolling reducers: any of ``mean``, ``count``, ``sum``, ``std``, ``max``.
    prefix:
        Column-name prefix; defaults to ``"{group_col}_{value_col}"``.

    Returns
    -------
    pd.DataFrame
        One column per ``(window, stat)`` pair, sharing the original index.

    Notes
    -----
    ``closed="left"`` guarantees the current row is excluded from its own window.
    ``std`` uses ddof=1 (pandas default); NaN is returned when fewer than 2
    observations exist in the window.
    """
    _validate_frame(frame, group_col, value_col, time_col)

    n = len(frame)
    if n == 0:
        return pd.DataFrame(index=frame.index)

    times = pd.DatetimeIndex(frame[time_col])
    unique_index = _make_unique_time_index(times, n)
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


# ---------------------------------------------------------------------------
# Velocity features
# ---------------------------------------------------------------------------


def add_velocity_features(
    frame: pd.DataFrame,
    *,
    group_col: str,
    time_col: str = "timestamp",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Seconds since the entity's previous transaction (past-only).

    Returns NaN for the entity's very first transaction.
    """
    if not frame[time_col].is_monotonic_increasing:
        raise ValueError("frame must be sorted by time_col")
    delta = frame.groupby(group_col, sort=False)[time_col].diff()
    column = out_col or f"{group_col}_seconds_since_prev"
    return pd.DataFrame({column: delta.dt.total_seconds()}, index=frame.index)


def add_transaction_count_in_window(
    frame: pd.DataFrame,
    *,
    group_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = ("1h", "24h", "7D"),
) -> pd.DataFrame:
    """Number of past transactions by the same entity in rolling windows.

    Uses a constant ``1`` series so we count events, not amounts.  This is a
    separate convenience wrapper — equivalent to ``past_rolling_aggregates`` with
    ``stats=("count",)`` but on a dummy value column, kept here for readability.
    """
    if time_col not in frame.columns:
        raise KeyError(f"Column {time_col!r} not found")
    if not frame[time_col].is_monotonic_increasing:
        raise ValueError("frame must be sorted by time_col")

    n = len(frame)
    times = pd.DatetimeIndex(frame[time_col])
    unique_index = _make_unique_time_index(times, n)
    ones = pd.Series(np.ones(n, dtype=float), index=unique_index)
    grouped = ones.groupby(frame[group_col].to_numpy(), sort=False)

    output = pd.DataFrame(index=frame.index)
    for window in windows:
        rolling_counts = grouped.rolling(window, closed="left").count()
        aligned = rolling_counts.droplevel(0).reindex(unique_index).fillna(0)
        output[f"{group_col}_txn_count_{window}"] = pd.Series(
            aligned.to_numpy(), index=frame.index
        )
    return output


# ---------------------------------------------------------------------------
# Extended: multi-entity aggregates (new — not in reference)
# ---------------------------------------------------------------------------


def add_merchant_aggregates(
    frame: pd.DataFrame,
    *,
    merchant_col: str,
    amount_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = ("24h", "7D"),
    stats: tuple[str, ...] = ("mean", "count", "std"),
    prefix: str = "merch",
) -> pd.DataFrame:
    """Past-only rolling aggregates grouped by merchant.

    Captures merchant-level fraud signals: a merchant seeing unusually many
    transactions or large amounts in the last 24 h is a risk signal.

    Parameters
    ----------
    merchant_col:
        Column identifying the merchant (e.g. ``"merchant"`` or ``"ProductCD"``).
    amount_col:
        Transaction amount column.
    """
    _validate_frame(frame, merchant_col, amount_col, time_col)
    return past_rolling_aggregates(
        frame,
        group_col=merchant_col,
        value_col=amount_col,
        time_col=time_col,
        windows=windows,
        stats=stats,
        prefix=prefix,
    )


def add_card_merchant_aggregates(
    frame: pd.DataFrame,
    *,
    card_col: str,
    merchant_col: str,
    amount_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = ("24h", "7D"),
    stats: tuple[str, ...] = ("mean", "count", "sum"),
    prefix: str = "card_merch",
) -> pd.DataFrame:
    """Past-only rolling aggregates for the card × merchant combination.

    This interaction captures the specific pattern of a card being used at the
    same merchant repeatedly — high signal for card-present fraud at a terminal.
    A temporary ``_combo`` column is used as the group key and dropped afterwards.
    """
    for col in (card_col, merchant_col, amount_col, time_col):
        if col not in frame.columns:
            raise KeyError(f"Column {col!r} not found in frame")
    if not frame[time_col].is_monotonic_increasing:
        raise ValueError("frame must be sorted by time_col")

    combo_col = "_combo_key"
    frame = frame.copy(deep=False)
    frame[combo_col] = (
        frame[card_col].astype(str) + "__" + frame[merchant_col].astype(str)
    )
    result = past_rolling_aggregates(
        frame,
        group_col=combo_col,
        value_col=amount_col,
        time_col=time_col,
        windows=windows,
        stats=stats,
        prefix=prefix,
    )
    # Drop the temporary column from frame view — it was added on a shallow copy
    return result


# ---------------------------------------------------------------------------
# Extended: amount-deviation features (new — not in reference)
# ---------------------------------------------------------------------------


def add_amount_deviation_features(
    frame: pd.DataFrame,
    *,
    card_col: str,
    amount_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = ("24h", "7D"),
) -> pd.DataFrame:
    """How unusual is this transaction amount compared to the card's past history?

    Computes two deviation features per window:

    * ``amount_zscore_{window}`` — z-score: ``(amount - past_mean) / past_std``.
      Undefined (NaN) when fewer than 2 past transactions exist.
    * ``amount_ratio_{window}`` — raw ratio: ``amount / past_mean``.
      NaN when past mean is 0 or missing.

    Both features use strictly past data (``closed="left"``).
    """
    _validate_frame(frame, card_col, amount_col, time_col)

    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    output = pd.DataFrame(index=frame.index)

    for window in windows:
        rolling_mean = past_rolling_aggregates(
            frame,
            group_col=card_col,
            value_col=amount_col,
            time_col=time_col,
            windows=(window,),
            stats=("mean",),
            prefix="_dev_mean",
        )[f"_dev_mean_mean_{window}"]

        rolling_std = past_rolling_aggregates(
            frame,
            group_col=card_col,
            value_col=amount_col,
            time_col=time_col,
            windows=(window,),
            stats=("std",),
            prefix="_dev_std",
        )[f"_dev_std_std_{window}"]

        output[f"amount_zscore_{window}"] = (amount - rolling_mean) / rolling_std.replace(0, np.nan)
        output[f"amount_ratio_{window}"] = amount / rolling_mean.replace(0, np.nan)

    return output


def add_high_spend_spike_flag(
    frame: pd.DataFrame,
    *,
    card_col: str,
    amount_col: str,
    time_col: str = "timestamp",
    window: str = "7D",
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Binary flag: 1 if amount > card's rolling (mean + n_std * std) over window.

    Useful as a simple, interpretable engineered feature for detecting abnormally
    large individual transactions relative to the card's recent spending history.
    """
    _validate_frame(frame, card_col, amount_col, time_col)

    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    rolling_mean = past_rolling_aggregates(
        frame,
        group_col=card_col,
        value_col=amount_col,
        time_col=time_col,
        windows=(window,),
        stats=("mean",),
        prefix="_spike_mean",
    )[f"_spike_mean_mean_{window}"]

    rolling_std = past_rolling_aggregates(
        frame,
        group_col=card_col,
        value_col=amount_col,
        time_col=time_col,
        windows=(window,),
        stats=("std",),
        prefix="_spike_std",
    )[f"_spike_std_std_{window}"]

    threshold = rolling_mean + n_std * rolling_std.fillna(0)
    flag = (amount > threshold).astype("Int8")
    # Where we have no history, default to 0 (no spike detected)
    flag = flag.fillna(0)
    return pd.DataFrame(
        {f"is_high_spend_spike_{window}": flag}, index=frame.index
    )


# ---------------------------------------------------------------------------
# Convenience: build all card-level aggregate features in one call
# ---------------------------------------------------------------------------


def build_card_features(
    frame: pd.DataFrame,
    *,
    card_col: str,
    amount_col: str,
    time_col: str = "timestamp",
    windows: tuple[str, ...] = ("1h", "24h", "7D"),
) -> pd.DataFrame:
    """Build all card-level past-only aggregate features in one call.

    Returns a DataFrame (original index) with:
    - Rolling mean / count / sum / std / max per window
    - Seconds since previous transaction
    - Transaction count per window
    - Amount z-score and ratio per window
    - High-spend spike flag (7D window)

    This is the recommended entry-point for the modeling pipeline.
    """
    parts = []

    # Core rolling aggregates
    parts.append(
        past_rolling_aggregates(
            frame,
            group_col=card_col,
            value_col=amount_col,
            time_col=time_col,
            windows=windows,
            stats=_EXTENDED_STATS,
            prefix=f"{card_col}_{amount_col}",
        )
    )

    # Velocity
    parts.append(
        add_velocity_features(frame, group_col=card_col, time_col=time_col)
    )

    # Transaction counts
    parts.append(
        add_transaction_count_in_window(
            frame, group_col=card_col, time_col=time_col, windows=windows
        )
    )

    # Amount deviation
    parts.append(
        add_amount_deviation_features(
            frame,
            card_col=card_col,
            amount_col=amount_col,
            time_col=time_col,
            windows=("24h", "7D"),
        )
    )

    # Spike flag
    parts.append(
        add_high_spend_spike_flag(
            frame,
            card_col=card_col,
            amount_col=amount_col,
            time_col=time_col,
            window="7D",
        )
    )

    return pd.concat(parts, axis=1)
