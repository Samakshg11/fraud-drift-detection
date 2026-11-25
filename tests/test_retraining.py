"""Tests for the Role 6 reference retraining strategies."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.retraining.cost_tracker import CostTracker
from src.retraining.strategies import (
    DriftTriggeredStrategy,
    PeriodicStrategy,
    SlidingWindowStrategy,
    StaticStrategy,
    WeightedStrategy,
)


def _kwargs() -> dict:
    return dict(
        model=LogisticRegression(max_iter=500, random_state=0),
        numeric_cols=["TransactionAmt", "card1", "addr1"],
        categorical_cols=["ProductCD", "P_emaildomain"],
        target_col="isFraud",
        scale_numeric=True,
    )


def test_static_never_refits(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=80)
    test = frame.loc[frame["month"] == frame["month"].max()]
    strategy = StaticStrategy(**_kwargs())
    strategy.fit(frame)
    first = strategy.predict_proba(test)
    strategy.fit(frame.loc[frame["month"] <= frame["month"].max() - 3])
    assert strategy.refit_count_ == 1
    np.testing.assert_allclose(first, strategy.predict_proba(test))


def test_periodic_refits_on_schedule(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=80)
    strategy = PeriodicStrategy(period_months=2, **_kwargs())
    months = sorted(frame["month"].unique())
    strategy.fit(frame.loc[frame["month"].isin(months[:6])])
    assert strategy.refit_count_ == 1
    strategy.fit(frame.loc[frame["month"].isin(months[:8])])
    assert strategy.refit_count_ == 2
    strategy.fit(frame.loc[frame["month"].isin(months[:9])])
    assert strategy.refit_count_ == 2


def test_sliding_window_limits_training_rows(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=80)
    strategy = SlidingWindowStrategy(window_months=3, **_kwargs())
    strategy.fit(frame)
    assert strategy.refit_count_ == 1
    assert strategy.last_train_rows_ == 3 * 80


def test_drift_triggered_refits_after_feedback(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=80)
    strategy = DriftTriggeredStrategy(
        method="adwin",
        detector_params={"delta": 0.05, "min_window_length": 5, "grace_period": 5},
        **_kwargs(),
    )
    strategy.fit(frame)
    assert strategy.refit_count_ == 1
    strategy.update(np.zeros(100, dtype=int), np.zeros(100))
    strategy.update(np.ones(100, dtype=int), np.zeros(100))  # all wrong
    strategy.fit(frame)
    assert strategy.refit_count_ == 2


def test_weighted_strategy_uses_recency_weights(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=80)
    strategy = WeightedStrategy(half_life_months=3, **_kwargs())
    strategy.fit(frame)
    assert strategy.refit_count_ == 1
    scores = strategy.predict_proba(frame.tail(80))
    assert np.isfinite(scores).all()


def test_cost_tracker_records_timing():
    tracker = CostTracker()
    result = tracker.track(strategy="static", month=1, train_rows=10, fn=lambda: "ok")
    assert result == "ok"
    frame = tracker.to_frame()
    assert len(frame) == 1
    assert frame.loc[0, "seconds"] >= 0
    assert frame.loc[0, "train_rows"] == 10
