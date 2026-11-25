"""Tests for chronological month assignment, splits and walk-forward folds."""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.splits import (
    add_month_index,
    derive_datetime,
    describe_splits,
    month_bounds,
    month_index_from_datetime,
    period_from_month_index,
    single_time_split,
    walk_forward_splits,
)


def test_derive_datetime_from_ieee_offset():
    values = pd.Series([0, 86_400], name="TransactionDT")
    out = derive_datetime(values, reference_date="2017-12-01", time_unit="seconds")
    assert out.iloc[0] == pd.Timestamp("2017-12-01")
    assert out.iloc[1] == pd.Timestamp("2017-12-02")
    assert out.name == "TransactionDT"


def test_month_index_crosses_year_boundary():
    dates = pd.Series(pd.to_datetime(["2018-12-31", "2019-01-01"]))
    idx = month_index_from_datetime(dates)
    assert idx.iloc[1] == idx.iloc[0] + 1
    assert period_from_month_index(int(idx.iloc[1])) == pd.Period("2019-01", freq="M")
    start, end = month_bounds(int(idx.iloc[1]))
    assert start == pd.Timestamp("2019-01-01")
    assert end == pd.Timestamp("2019-02-01")


def test_add_month_index_does_not_mutate_input(make_synthetic):
    frame = make_synthetic(with_timestamps=False)
    out = add_month_index(
        frame,
        time_col="TransactionDT",
        reference_date="2017-12-01",
        time_unit="seconds",
    )
    assert "month" not in frame.columns
    assert "month" in out.columns
    assert out["month"].nunique() == 12


def test_single_split_is_chronological_and_respects_embargo(make_synthetic):
    frame = make_synthetic(months=10)
    split = single_time_split(
        frame,
        time_col="timestamp",
        train_months=3,
        test_months=2,
        gap_days=10,
    )
    assert split.train_end < split.test_start
    assert len(split.train_months) == 3
    assert len(split.test_months) == 2
    month_start, _ = month_bounds(split.test_months[0])
    assert split.test_start >= month_start + pd.Timedelta(days=10)


def test_walk_forward_expanding_grows_and_test_blocks_disjoint(make_synthetic):
    frame = make_synthetic(months=12)
    splits = list(
        walk_forward_splits(
            frame,
            time_col="timestamp",
            train_months=3,
            test_months=2,
        )
    )
    # 12 observed months, 3 train + 2 test => anchors 2,4,6,8 (0-based).
    assert len(splits) == 4
    lengths = [len(s.train_months) for s in splits]
    assert lengths == [3, 5, 7, 9]  # expanding window anchored at first month

    test_indices = [set(s.test_idx) for s in splits]
    for i in range(len(test_indices) - 1):
        assert test_indices[i].isdisjoint(test_indices[i + 1])

    for split in splits:
        train_idx, test_idx = split
        assert frame.loc[train_idx, "timestamp"].max() < frame.loc[test_idx, "timestamp"].min()
        assert len(train_idx.intersection(test_idx)) == 0


def test_walk_forward_sliding_caps_training_window(make_synthetic):
    frame = make_synthetic(months=12)
    splits = list(
        walk_forward_splits(
            frame,
            time_col="timestamp",
            train_months=6,
            test_months=2,
            scheme="sliding",
            sliding_window_months=3,
        )
    )
    # 12 months, 6 train + 2 test => anchors 5,7,9 (0-based).
    assert len(splits) == 3
    assert all(len(s.train_months) == 3 for s in splits)
    first_start = splits[0].train_months[0]
    assert [s.train_months[0] for s in splits] == [
        first_start + 2 * i for i in range(len(splits))
    ]


def test_large_embargo_can_skip_empty_folds(make_synthetic):
    frame = make_synthetic(months=6, n_per_month=10)
    splits = list(
        walk_forward_splits(
            frame,
            time_col="timestamp",
            train_months=2,
            test_months=1,
            gap_days=200,  # removes most of each one-month test block
        )
    )
    assert all(s.test_start > s.train_end for s in splits)


def test_describe_splits_has_metadata(make_synthetic):
    frame = make_synthetic(months=8)
    splits = list(
        walk_forward_splits(frame, time_col="timestamp", train_months=2, test_months=1)
    )
    table = describe_splits(splits)
    assert table["fold"].tolist() == list(range(len(splits)))
    assert (table["train_rows"] > 0).all()


def test_requires_datetime_time_column(make_synthetic):
    frame = make_synthetic(with_timestamps=False).assign(month=0)
    with pytest.raises(TypeError, match="datetime64"):
        single_time_split(
            frame,
            time_col="TransactionDT",
            train_months=2,
            test_months=1,
        )
