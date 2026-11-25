"""Tests for strictly-past-only target and frequency encoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.encoders import FrequencyEncoder, TimeAwareTargetEncoder


def _target_frame() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "cat": ["a", "a", "a", "b"],
            "timestamp": pd.to_datetime(
                ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
            ),
        }
    )
    return X, np.array([1, 0, 1, 0])


def test_target_encoder_uses_only_strictly_earlier_rows():
    X, y = _target_frame()
    out = TimeAwareTargetEncoder(
        ["cat"], time_col="timestamp", smoothing=1.0, prior=0.0
    ).fit_transform(X, y)
    # Shrinkage is toward the PAST global mean (prior at t=0 is the fallback 0):
    # t1 -> 0; t2 -> (1 + 1*1)/(1+1)=1; t3 -> (1 + 1*0.5)/(2+1)=0.5;
    # t4 is category "b" with no past -> its past-global mean 2/3, times smoothing.
    np.testing.assert_allclose(out["cat"].to_numpy(), [0.0, 1.0, 0.5, 2 / 3])


def test_future_label_cannot_change_past_encodings():
    X, y = _target_frame()
    encoder = TimeAwareTargetEncoder(["cat"], time_col="timestamp", smoothing=1.0)
    before = encoder.fit_transform(X, y)
    changed = y.copy()
    changed[-1] = 1  # change only the FUTURE row's label
    after = encoder.fit_transform(X, changed)
    np.testing.assert_allclose(before["cat"].to_numpy()[:3], after["cat"].to_numpy()[:3])


def test_same_timestamp_rows_do_not_influence_each_other():
    X = pd.DataFrame(
        {
            "cat": ["a", "a"],
            "timestamp": pd.to_datetime(["2019-01-01", "2019-01-01"]),
        }
    )
    out = TimeAwareTargetEncoder(
        ["cat"], time_col="timestamp", smoothing=1.0, prior=0.0
    ).fit_transform(X, np.array([1, 0]))
    np.testing.assert_allclose(out["cat"].to_numpy(), [0.0, 0.0])


def test_transform_uses_frozen_train_mapping_and_never_refits():
    X, y = _target_frame()
    encoder = TimeAwareTargetEncoder(["cat"], time_col="timestamp", smoothing=1.0).fit(
        X, y
    )
    mapping_before = encoder.mapping_["cat"].copy()

    test = pd.DataFrame(
        {
            "cat": ["a", "c"],  # "c" unseen in training
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    out = encoder.transform(test)
    assert out.loc[0, "cat"] == pytest.approx(encoder.mapping_["cat"]["a"])
    assert out.loc[1, "cat"] == pytest.approx(encoder.prior_)
    pd.testing.assert_series_equal(encoder.mapping_["cat"], mapping_before)


def test_expanding_values_differ_from_frozen_full_train_mapping():
    X, y = _target_frame()
    encoder = TimeAwareTargetEncoder(["cat"], time_col="timestamp", smoothing=1.0)
    expanding = encoder.fit_transform(X, y)["cat"].to_numpy()
    frozen = encoder.transform(X)["cat"].to_numpy()
    assert not np.allclose(expanding, frozen)


def test_frequency_encoder_past_counts_then_frozen_counts():
    X = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "a"],
            "timestamp": pd.to_datetime(
                ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
            ),
        }
    )
    encoder = FrequencyEncoder(["cat"], time_col="timestamp")
    past = encoder.fit_transform(X)["cat"].to_numpy()
    np.testing.assert_allclose(past, [0.0, 1.0, 0.0, 2.0])

    future = pd.DataFrame(
        {
            "cat": ["a", "unknown"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    out = encoder.transform(future)["cat"].to_numpy()
    np.testing.assert_allclose(out, [3.0, 0.0])
    assert "unknown" not in encoder.counts_["cat"].index


def test_frequency_encoder_normalized_share():
    X = pd.DataFrame({"cat": ["a", "a", "b", "a"]})
    # The denominator is the number of STRICTLY earlier rows (1 and 3 here),
    # never the full frame size -- using 4 would leak the future into row 1.
    out = FrequencyEncoder(["cat"], normalize=True).fit_transform(X)
    np.testing.assert_allclose(out["cat"].to_numpy(), [0.0, 1.0, 0.0, 2 / 3])


def test_missing_categories_are_grouped_not_dropped():
    X = pd.DataFrame({"cat": [None, "a", None], "timestamp": range(3)})
    out = FrequencyEncoder(["cat"]).fit_transform(X)
    assert out["cat"].notna().all()
    assert out.loc[2, "cat"] == 1.0
