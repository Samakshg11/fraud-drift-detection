"""Leakage-correctness tests for src/features/ (Role 3 — Harsh Kumar).

These tests assert the core anti-leakage guarantee of every aggregate feature:

    For every row i, no feature value at row i may depend on any
    observation at time >= timestamp[i].

The tests use synthetic DataFrames with known values so the expected outputs can
be computed by hand and asserted exactly.

Run with:
    pytest tests/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.aggregates import (
    past_rolling_aggregates,
    add_velocity_features,
    add_transaction_count_in_window,
    add_amount_deviation_features,
    add_high_spend_spike_flag,
    build_card_features,
)
from src.features.time_features import add_time_features
from src.features.amount_features import (
    AmountFeaturizer,
    build_amount_features,
    log1p_amount,
    round_amount_flag,
    cents_component,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_frame() -> pd.DataFrame:
    """Four transactions for a single card, 1 hour apart."""
    return pd.DataFrame(
        {
            "card_id": ["A", "A", "A", "A"],
            "merchant": ["M1", "M2", "M1", "M3"],
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01 10:00",
                    "2021-01-01 11:00",
                    "2021-01-01 12:00",
                    "2021-01-01 13:00",
                ]
            ),
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture()
def two_card_frame() -> pd.DataFrame:
    """Two cards, alternating transactions, to test isolation between groups."""
    return pd.DataFrame(
        {
            "card_id": ["A", "B", "A", "B", "A"],
            "merchant": ["M1", "M1", "M2", "M2", "M1"],
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01 09:00",
                    "2021-01-01 10:00",
                    "2021-01-01 11:00",
                    "2021-01-01 12:00",
                    "2021-01-01 13:00",
                ]
            ),
            "amount": [5.0, 50.0, 15.0, 100.0, 25.0],
        }
    )


# ---------------------------------------------------------------------------
# 1. First-row leakage: row 0 must have NaN (no past data exists)
# ---------------------------------------------------------------------------


class TestFirstRowIsNaN:
    def test_rolling_mean_first_row_nan(self, simple_frame):
        """The first transaction must have NaN rolling mean — no past data."""
        result = past_rolling_aggregates(
            simple_frame,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("mean",),
        )
        assert np.isnan(result.iloc[0]["card_id_amount_mean_24h"]), (
            "First row's rolling mean must be NaN (no past transactions)"
        )

    def test_rolling_count_first_row_zero(self, simple_frame):
        """Count for the first row must be 0 (no past transactions)."""
        result = past_rolling_aggregates(
            simple_frame,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("count",),
        )
        assert result.iloc[0]["card_id_amount_count_24h"] == 0, (
            "First row's rolling count must be 0 (no past transactions)"
        )

    def test_velocity_first_row_nan(self, simple_frame):
        """Seconds-since-previous for first transaction must be NaN."""
        result = add_velocity_features(
            simple_frame, group_col="card_id", time_col="timestamp"
        )
        assert np.isnan(result.iloc[0].iloc[0]), (
            "First row velocity must be NaN (no previous transaction)"
        )


# ---------------------------------------------------------------------------
# 2. Core leakage test: feature at row i uses only rows 0..i-1
# ---------------------------------------------------------------------------


class TestNoFutureData:
    def test_rolling_mean_excludes_current_row(self, simple_frame):
        """Rolling mean at row i must equal mean(amounts[0..i-1])."""
        result = past_rolling_aggregates(
            simple_frame,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("mean",),
        )
        col = "card_id_amount_mean_24h"
        # Row 1: past = [10], mean = 10
        assert result.iloc[1][col] == pytest.approx(10.0), (
            "Row 1 mean should be 10 (only row 0 is in the past)"
        )
        # Row 2: past = [10, 20], mean = 15
        assert result.iloc[2][col] == pytest.approx(15.0), (
            "Row 2 mean should be 15 (rows 0 and 1 are in the past)"
        )
        # Row 3: past = [10, 20, 30], mean = 20
        assert result.iloc[3][col] == pytest.approx(20.0), (
            "Row 3 mean should be 20 (rows 0, 1, 2 are in the past)"
        )

    def test_rolling_count_excludes_current_row(self, simple_frame):
        """Rolling count at row i must equal i (number of past rows for entity A)."""
        result = past_rolling_aggregates(
            simple_frame,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("count",),
        )
        col = "card_id_amount_count_24h"
        expected = [0, 1, 2, 3]
        for i, exp in enumerate(expected):
            assert result.iloc[i][col] == pytest.approx(exp), (
                f"Row {i} count should be {exp}"
            )

    def test_rolling_sum_excludes_current_row(self, simple_frame):
        """Rolling sum at row i = sum(amounts[0..i-1])."""
        result = past_rolling_aggregates(
            simple_frame,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("sum",),
        )
        col = "card_id_amount_sum_24h"
        expected = [np.nan, 10.0, 30.0, 60.0]
        for i, exp in enumerate(expected):
            if np.isnan(exp):
                assert np.isnan(result.iloc[i][col])
            else:
                assert result.iloc[i][col] == pytest.approx(exp), (
                    f"Row {i} sum should be {exp}"
                )

    def test_velocity_excludes_current_row(self, simple_frame):
        """Velocity = seconds since previous transaction (past-only)."""
        result = add_velocity_features(
            simple_frame, group_col="card_id", time_col="timestamp"
        )
        col = result.columns[0]
        # Each row is 3600 seconds (1 hour) apart
        assert np.isnan(result.iloc[0][col])            # first row
        assert result.iloc[1][col] == pytest.approx(3600.0)
        assert result.iloc[2][col] == pytest.approx(3600.0)
        assert result.iloc[3][col] == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# 3. Group isolation: card A's features must not include card B's data
# ---------------------------------------------------------------------------


class TestGroupIsolation:
    def test_different_cards_isolated(self, two_card_frame):
        """Card A and card B rolling aggregates must be computed independently."""
        result = past_rolling_aggregates(
            two_card_frame,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("mean",),
        )
        col = "card_id_amount_mean_24h"

        # Row 0 (card A, t=09:00): no past — NaN
        assert np.isnan(result.iloc[0][col])

        # Row 1 (card B, t=10:00): card B has no past — NaN (card A's data
        # must NOT bleed into card B's aggregate)
        assert np.isnan(result.iloc[1][col]), (
            "Card B first row must be NaN; card A's transactions should not bleed in"
        )

        # Row 2 (card A, t=11:00): past for card A = [5], mean = 5
        assert result.iloc[2][col] == pytest.approx(5.0), (
            "Card A row 2 mean should be 5 (only row 0 from card A is in the past)"
        )

        # Row 3 (card B, t=12:00): past for card B = [50], mean = 50
        assert result.iloc[3][col] == pytest.approx(50.0), (
            "Card B row 3 mean should be 50 (only row 1 from card B is in the past)"
        )


# ---------------------------------------------------------------------------
# 4. Sorted order requirement
# ---------------------------------------------------------------------------


class TestSortedOrderEnforced:
    def test_unsorted_raises(self):
        """Passing an unsorted frame must raise ValueError."""
        df = pd.DataFrame(
            {
                "card_id": ["A", "A"],
                "amount": [10.0, 20.0],
                "timestamp": pd.to_datetime(
                    ["2021-01-02 10:00", "2021-01-01 10:00"]  # reversed!
                ),
            }
        )
        with pytest.raises(ValueError, match="sorted"):
            past_rolling_aggregates(
                df,
                group_col="card_id",
                value_col="amount",
                time_col="timestamp",
                windows=("24h",),
                stats=("mean",),
            )

    def test_unsorted_velocity_raises(self):
        """Velocity function must also raise on unsorted input."""
        df = pd.DataFrame(
            {
                "card_id": ["A", "A"],
                "timestamp": pd.to_datetime(
                    ["2021-01-02", "2021-01-01"]   # reversed
                ),
                "amount": [1.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="sorted"):
            add_velocity_features(df, group_col="card_id", time_col="timestamp")


# ---------------------------------------------------------------------------
# 5. Amount deviation features leakage test
# ---------------------------------------------------------------------------


class TestAmountDeviation:
    def test_zscore_first_row_nan(self, simple_frame):
        """Z-score for first row must be NaN (no past data)."""
        result = add_amount_deviation_features(
            simple_frame,
            card_col="card_id",
            amount_col="amount",
            time_col="timestamp",
            windows=("24h",),
        )
        assert np.isnan(result.iloc[0]["amount_zscore_24h"]), (
            "Z-score for first row should be NaN"
        )

    def test_zscore_excludes_current_amount(self, simple_frame):
        """Z-score at row 2 should use only rows 0,1 for mean/std."""
        result = add_amount_deviation_features(
            simple_frame,
            card_col="card_id",
            amount_col="amount",
            time_col="timestamp",
            windows=("24h",),
        )
        # Row 2: past amounts = [10, 20], mean=15, std=7.07, amount=30
        # z = (30 - 15) / 7.07 ≈ 2.12
        zscore_row2 = result.iloc[2]["amount_zscore_24h"]
        expected = (30.0 - 15.0) / np.std([10.0, 20.0], ddof=1)
        assert zscore_row2 == pytest.approx(expected, rel=1e-3), (
            f"Z-score at row 2 should be ≈ {expected:.3f}, got {zscore_row2:.3f}"
        )


# ---------------------------------------------------------------------------
# 6. AmountFeaturizer: fit on train only, transform test without re-fitting
# ---------------------------------------------------------------------------


class TestAmountFeaturizer:
    def test_fit_transform_shape(self, simple_frame):
        """Fit on frame, transform — output must have same number of rows."""
        af = AmountFeaturizer(amount_col="amount")
        result = af.fit_transform(simple_frame)
        assert len(result) == len(simple_frame)

    def test_transform_before_fit_raises(self, simple_frame):
        """Calling .transform() before .fit() must raise RuntimeError."""
        af = AmountFeaturizer(amount_col="amount")
        with pytest.raises(RuntimeError, match="fit"):
            af.transform(simple_frame)

    def test_global_zscore_uses_train_stats(self):
        """Global z-score must use train-set mean/std, not test-set statistics."""
        train = pd.DataFrame({"amount": [10.0, 20.0, 30.0]})
        test = pd.DataFrame({"amount": [100.0, 200.0]})

        af = AmountFeaturizer(amount_col="amount")
        af.fit(train)

        train_mean = 20.0
        train_std = float(np.std([10.0, 20.0, 30.0], ddof=1))

        result = af.transform(test)
        expected_z0 = (100.0 - train_mean) / train_std
        expected_z1 = (200.0 - train_mean) / train_std

        assert result.iloc[0]["amount_global_zscore"] == pytest.approx(expected_z0, rel=1e-4)
        assert result.iloc[1]["amount_global_zscore"] == pytest.approx(expected_z1, rel=1e-4)

    def test_percentile_flags_use_train_thresholds(self):
        """is_above_p90 flag must be based on train-set 90th percentile."""
        train = pd.DataFrame({"amount": list(range(1, 101))})   # 1..100, p90=90
        test = pd.DataFrame({"amount": [89.0, 91.0]})

        af = AmountFeaturizer(amount_col="amount").fit(train)
        result = af.transform(test)

        assert result.iloc[0]["is_above_p90"] == 0, "89 is not above train p90=90"
        assert result.iloc[1]["is_above_p90"] == 1, "91 is above train p90=90"


# ---------------------------------------------------------------------------
# 7. Stateless amount features
# ---------------------------------------------------------------------------


class TestStatelessAmountFeatures:
    def test_log1p_zero_amount(self):
        """log1p(0) = 0.0, not NaN."""
        df = pd.DataFrame({"TransactionAmt": [0.0, 1.0, 10.0]})
        result = log1p_amount(df)
        assert result.iloc[0]["TransactionAmt_log1p"] == pytest.approx(0.0)

    def test_round_flag_detects_integer_amounts(self):
        """Integer amounts must be flagged as round."""
        df = pd.DataFrame({"TransactionAmt": [10.0, 10.5, 100.0, 0.99]})
        result = round_amount_flag(df)
        expected = [1, 0, 1, 0]
        for i, exp in enumerate(expected):
            assert result.iloc[i]["TransactionAmt_is_round"] == exp

    def test_cents_component_range(self):
        """Cents component must be in [0.00, 0.99]."""
        df = pd.DataFrame({"TransactionAmt": [19.99, 100.0, 50.50, 0.01]})
        result = cents_component(df)
        cents = result["TransactionAmt_cents"]
        assert (cents >= 0.0).all() and (cents < 1.0).all()


# ---------------------------------------------------------------------------
# 8. Time features — no leakage (calendar features are always row-local)
# ---------------------------------------------------------------------------


class TestTimeFeatures:
    def test_time_features_shape(self, simple_frame):
        """add_time_features must return one row per input row."""
        result = add_time_features(simple_frame, time_col="timestamp")
        assert len(result) == len(simple_frame)

    def test_is_night_flag(self):
        """is_night must be 1 for 00:00–05:59, 0 otherwise."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2021-01-01 02:30", "2021-01-01 09:00", "2021-01-01 23:59"]
                )
            }
        )
        result = add_time_features(df, time_col="timestamp")
        assert result.iloc[0]["is_night"] == 1
        assert result.iloc[1]["is_night"] == 0
        assert result.iloc[2]["is_night"] == 0

    def test_is_weekend_flag(self):
        """is_weekend must be 1 on Sat/Sun, 0 on weekdays."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2021-01-04",   # Monday
                        "2021-01-09",   # Saturday
                        "2021-01-10",   # Sunday
                    ]
                )
            }
        )
        result = add_time_features(df, time_col="timestamp")
        assert result.iloc[0]["is_weekend"] == 0, "Monday should not be weekend"
        assert result.iloc[1]["is_weekend"] == 1, "Saturday should be weekend"
        assert result.iloc[2]["is_weekend"] == 1, "Sunday should be weekend"

    def test_is_business_hours(self):
        """is_business_hours: 09:00–17:59, Mon–Fri only."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2021-01-04 10:00",   # Monday 10am — YES
                        "2021-01-04 20:00",   # Monday 8pm — NO
                        "2021-01-09 10:00",   # Saturday 10am — NO
                    ]
                )
            }
        )
        result = add_time_features(df, time_col="timestamp")
        assert result.iloc[0]["is_business_hours"] == 1
        assert result.iloc[1]["is_business_hours"] == 0
        assert result.iloc[2]["is_business_hours"] == 0

    def test_missing_time_col_raises(self):
        """Missing time column must raise KeyError."""
        df = pd.DataFrame({"amount": [10.0]})
        with pytest.raises(KeyError):
            add_time_features(df, time_col="timestamp")


# ---------------------------------------------------------------------------
# 9. Transaction count per window
# ---------------------------------------------------------------------------


class TestTransactionCount:
    def test_count_first_row_zero(self, simple_frame):
        """Transaction count for first row must be 0 (no past events)."""
        result = add_transaction_count_in_window(
            simple_frame,
            group_col="card_id",
            time_col="timestamp",
            windows=("24h",),
        )
        assert result.iloc[0]["card_id_txn_count_24h"] == 0

    def test_count_increases_monotonically(self, simple_frame):
        """With all transactions within 24h window, count should be 0,1,2,3."""
        result = add_transaction_count_in_window(
            simple_frame,
            group_col="card_id",
            time_col="timestamp",
            windows=("24h",),
        )
        col = "card_id_txn_count_24h"
        counts = result[col].tolist()
        assert counts == pytest.approx([0, 1, 2, 3]), (
            f"Expected [0,1,2,3] got {counts}"
        )


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_frame_returns_empty(self):
        """Empty input must return an empty DataFrame without error."""
        df = pd.DataFrame(
            columns=["card_id", "amount", "timestamp"]
        ).astype({"timestamp": "datetime64[ns]"})
        result = past_rolling_aggregates(
            df,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
        )
        assert len(result) == 0

    def test_single_row_no_past(self):
        """A single-row frame must produce NaN mean and 0 count."""
        df = pd.DataFrame(
            {
                "card_id": ["A"],
                "amount": [50.0],
                "timestamp": pd.to_datetime(["2021-01-01"]),
            }
        )
        result = past_rolling_aggregates(
            df,
            group_col="card_id",
            value_col="amount",
            time_col="timestamp",
            windows=("24h",),
            stats=("mean", "count"),
        )
        assert np.isnan(result.iloc[0]["card_id_amount_mean_24h"])
        assert result.iloc[0]["card_id_amount_count_24h"] == 0

    def test_missing_column_raises_key_error(self, simple_frame):
        """Passing a non-existent column must raise KeyError."""
        with pytest.raises(KeyError):
            past_rolling_aggregates(
                simple_frame,
                group_col="nonexistent_col",
                value_col="amount",
                time_col="timestamp",
            )

    def test_non_datetime_col_raises_type_error(self, simple_frame):
        """A non-datetime time column must raise TypeError."""
        bad = simple_frame.copy()
        bad["timestamp"] = bad["timestamp"].astype(str)   # string, not datetime
        with pytest.raises(TypeError):
            past_rolling_aggregates(
                bad,
                group_col="card_id",
                value_col="amount",
                time_col="timestamp",
            )
