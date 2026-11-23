"""Tests for the deliberately leaky baseline and its leakage-free counterpart."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.pipeline.leaky_baseline import (
    add_global_aggregates,
    leaky_random_split,
    run_leakage_free_baseline,
    run_leaky_baseline,
)


def _model():
    return LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)


def _common_kwargs():
    return dict(
        target_col="isFraud",
        time_col="timestamp",
        numeric_cols=["TransactionAmt", "card1", "addr1"],
        categorical_cols=["ProductCD", "P_emaildomain"],
        target_encoded_cols=["ProductCD"],
        frequency_encoded_cols=["P_emaildomain"],
        model=_model(),
        scale_numeric=True,
    )


def test_random_split_mixes_past_and_future(make_synthetic):
    frame = make_synthetic(months=12)
    _, test_idx = leaky_random_split(
        frame, target_col="isFraud", test_size=0.25, seed=7
    )
    test_months = frame.loc[test_idx, "month"].unique()
    assert len(test_months) > 1
    assert test_months.min() == frame["month"].min()
    assert test_months.max() == frame["month"].max()


def test_global_aggregates_use_full_timeline(make_synthetic):
    frame = make_synthetic(months=6)
    out, columns = add_global_aggregates(
        frame, group_col="ProductCD", value_col="TransactionAmt"
    )
    expected = frame.groupby("ProductCD")["TransactionAmt"].transform("mean")
    np.testing.assert_allclose(out[columns[0]].to_numpy(), expected.to_numpy())
    assert (out[columns[1]] > 0).all()


def test_both_baselines_produce_predictions(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=120)
    leaky = run_leaky_baseline(frame, seed=7, **_common_kwargs())
    clean = run_leakage_free_baseline(
        frame,
        **_common_kwargs(),
        month_col="month",
        train_months=6,
        test_months=2,
    )
    for result in (leaky, clean):
        assert result.n_rows > 0
        assert set(np.unique(result.y_true)).issubset({0, 1})
        assert ((result.y_score >= 0) & (result.y_score <= 1)).all()
    assert len(clean.fold_summaries) >= 2
    for summary in clean.fold_summaries:
        assert summary["train_end"] < summary["test_start"]


def test_leaky_baseline_is_deterministic(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=80)
    first = run_leaky_baseline(frame, seed=11, **_common_kwargs())
    second = run_leaky_baseline(frame, seed=11, **_common_kwargs())
    np.testing.assert_array_equal(first.y_true, second.y_true)
    np.testing.assert_allclose(first.y_score, second.y_score)
