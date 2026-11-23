"""Tests for the Role 3 reference feature implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.aggregates import add_velocity_features, past_rolling_aggregates
from src.features.time_features import add_time_features


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-01-01 10:00",
                    "2019-01-01 11:00",
                    "2019-01-01 12:00",
                    "2019-01-02 10:00",
                    "2019-02-01 10:00",
                ]
            ),
            "card": ["a", "a", "b", "a", "a"],
            "amt": [10.0, 20.0, 100.0, 30.0, 40.0],
        }
    )
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def test_past_rolling_aggregates_exclude_current_row():
    out = past_rolling_aggregates(
        _frame(),
        group_col="card",
        value_col="amt",
        windows=("24h", "7D"),
        stats=("mean", "count"),
    )
    assert out.loc[0, "card_amt_count_24h"] == 0
    assert np.isnan(out.loc[0, "card_amt_mean_24h"])
    assert out.loc[1, "card_amt_count_24h"] == 1
    assert out.loc[1, "card_amt_mean_24h"] == 10.0
    assert out.loc[2, "card_amt_count_24h"] == 0  # different group
    # Jan-02 10:00: 11:00 yesterday is within 24h; 10:00 yesterday is not.
    assert out.loc[3, "card_amt_count_24h"] == 1
    assert out.loc[3, "card_amt_mean_24h"] == 20.0
    assert out.loc[3, "card_amt_count_7D"] == 2
    assert out.loc[3, "card_amt_mean_7D"] == 15.0
    assert out.loc[4, "card_amt_count_7D"] == 0  # 30 days later


def test_future_rows_cannot_change_past_features():
    frame = _frame()
    before = past_rolling_aggregates(
        frame, group_col="card", value_col="amt", windows=("7D",)
    )
    changed = frame.copy()
    changed.loc[4, "amt"] = 999.0  # last row only
    after = past_rolling_aggregates(
        changed, group_col="card", value_col="amt", windows=("7D",)
    )
    pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])


def test_unsorted_frame_is_rejected():
    frame = _frame().sample(frac=1, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        past_rolling_aggregates(frame, group_col="card", value_col="amt")


def test_velocity_is_past_only_and_group_scoped():
    out = add_velocity_features(_frame(), group_col="card")
    column = "card_seconds_since_previous"
    assert np.isnan(out.loc[0, column])
    assert out.loc[1, column] == 3600.0
    assert np.isnan(out.loc[2, column])  # first transaction of card b
    assert out.loc[3, column] == 23 * 3600.0


def test_time_features_are_deterministic_and_bounded():
    features = add_time_features(_frame())
    assert features.loc[0, "hour"] == 10
    assert features.loc[0, "dayofweek"] == pd.Timestamp("2019-01-01").day_of_week
    assert features.loc[0, "is_weekend"] == 0
    assert not features.isna().any().any()
    cyclic = [c for c in features.columns if c.endswith(("_sin", "_cos"))]
    assert ((features[cyclic] >= -1) & (features[cyclic] <= 1)).all().all()
