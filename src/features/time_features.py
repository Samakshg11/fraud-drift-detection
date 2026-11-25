"""REFERENCE IMPLEMENTATION for Role 3 (Feature Engineering) -- safe to replace.

Calendar features depend only on a row's own timestamp, so they can be computed
on all rows without any temporal leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(
    frame: pd.DataFrame, *, time_col: str = "timestamp", prefix: str = ""
) -> pd.DataFrame:
    """Return cyclic calendar features for each transaction."""
    times = pd.to_datetime(frame[time_col])
    hour = times.dt.hour.to_numpy()
    dayofweek = times.dt.day_of_week.to_numpy()
    month = times.dt.month.to_numpy()

    features = pd.DataFrame(index=frame.index)
    stem = f"{prefix}_" if prefix else ""
    features[f"{stem}hour"] = hour
    features[f"{stem}dayofweek"] = dayofweek
    features[f"{stem}day_of_month"] = times.dt.day.to_numpy()
    features[f"{stem}month"] = month
    features[f"{stem}is_weekend"] = (dayofweek >= 5).astype(int)
    features[f"{stem}hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features[f"{stem}hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features[f"{stem}dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    features[f"{stem}dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    features[f"{stem}month_sin"] = np.sin(2 * np.pi * month / 12)
    features[f"{stem}month_cos"] = np.cos(2 * np.pi * month / 12)
    return features
