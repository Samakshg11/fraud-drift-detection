"""Time-based features for fraud detection (Role 3 — Harsh).

Calendar features depend only on a row's own timestamp, so they can be computed
on the full dataset without any temporal leakage.

Extended relative to the reference skeleton
-------------------------------------------
* ``is_night`` — 00:00–05:59, known high-fraud window.
* ``is_business_hours`` — 09:00–17:59, Mon–Fri.
* ``quarter`` and ``is_month_end`` — month-end spending spikes are known fraud
  patterns in banking data.
* ``is_holiday_season`` — November–December (retail fraud peaks).
* ``days_since_epoch`` — a monotonically increasing numeric time index useful
  for drift-aware models that need to treat time as a covariate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(
    frame: pd.DataFrame, *, time_col: str = "timestamp", prefix: str = ""
) -> pd.DataFrame:
    """Return cyclic and categorical calendar features for each transaction.

    Parameters
    ----------
    frame:
        Input DataFrame.  Must contain ``time_col``.
    time_col:
        Name of the datetime column.
    prefix:
        Optional string prepended to every output column name.

    Returns
    -------
    pd.DataFrame
        New columns only (original index preserved).  Join with the input
        frame using ``pd.concat([frame, add_time_features(frame)], axis=1)``.
    """
    if time_col not in frame.columns:
        raise KeyError(f"Column {time_col!r} not found")

    times = pd.to_datetime(frame[time_col])
    hour = times.dt.hour.to_numpy()
    dayofweek = times.dt.day_of_week.to_numpy()   # 0=Mon … 6=Sun
    month = times.dt.month.to_numpy()

    stem = f"{prefix}_" if prefix else ""
    features = pd.DataFrame(index=frame.index)

    # ------------------------------------------------------------------
    # Raw calendar fields
    # ------------------------------------------------------------------
    features[f"{stem}hour"] = hour
    features[f"{stem}dayofweek"] = dayofweek
    features[f"{stem}day_of_month"] = times.dt.day.to_numpy()
    features[f"{stem}month"] = month
    features[f"{stem}quarter"] = times.dt.quarter.to_numpy()

    # ------------------------------------------------------------------
    # Cyclic encodings (sine / cosine) — avoids discontinuity at midnight
    # ------------------------------------------------------------------
    features[f"{stem}hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features[f"{stem}hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features[f"{stem}dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    features[f"{stem}dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    features[f"{stem}month_sin"] = np.sin(2 * np.pi * month / 12)
    features[f"{stem}month_cos"] = np.cos(2 * np.pi * month / 12)

    # ------------------------------------------------------------------
    # Binary flags — fraud domain knowledge
    # ------------------------------------------------------------------
    features[f"{stem}is_weekend"] = (dayofweek >= 5).astype(int)

    # Night: 00:00–05:59.  Fraud is disproportionately common at night because
    # legitimate cardholders are unlikely to notice a notification.
    features[f"{stem}is_night"] = ((hour >= 0) & (hour < 6)).astype(int)

    # Business hours: 09:00–17:59, Monday–Friday.
    features[f"{stem}is_business_hours"] = (
        (hour >= 9) & (hour < 18) & (dayofweek < 5)
    ).astype(int)

    # Month-end: last 3 days of the month.  Legitimate high-spend period that
    # fraudsters exploit (fewer anomaly alerts among many real transactions).
    features[f"{stem}is_month_end"] = (
        times.dt.is_month_end | (times.dt.day >= 28)
    ).astype(int)

    # Holiday season: November (11) and December (12).
    features[f"{stem}is_holiday_season"] = (month >= 11).astype(int)

    # ------------------------------------------------------------------
    # Monotonic time covariate — useful for drift-aware models
    # ------------------------------------------------------------------
    epoch = pd.Timestamp("2017-01-01")           # arbitrary stable reference
    features[f"{stem}days_since_epoch"] = (times - epoch).dt.days.to_numpy()

    return features


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def add_all_time_features(
    frame: pd.DataFrame, *, time_col: str = "timestamp"
) -> pd.DataFrame:
    """Add time features directly to the input frame (returns a copy).

    Convenience wrapper that concatenates the original frame with the output
    of :func:`add_time_features`.
    """
    return pd.concat([frame, add_time_features(frame, time_col=time_col)], axis=1)
