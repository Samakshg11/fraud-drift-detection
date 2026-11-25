"""Tests for the Role 1 reference cleaning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.clean import build_timestamp, downcast_numeric


def test_downcast_preserves_large_timestamp_precision():
    frame = pd.DataFrame(
        {
            "small": np.array([1.5, 2.5], dtype="float64"),
            "TransactionDT": np.array([63_000_000.0, 63_086_400.0]),
            "id": np.array([1, 2], dtype="int64"),
        }
    )
    out = downcast_numeric(frame)
    assert out["small"].dtype == np.float32
    assert out["TransactionDT"].dtype == np.float64
    assert out["id"].dtype != np.int64  # downcast to smallest int
    np.testing.assert_array_equal(
        out["TransactionDT"].to_numpy(), frame["TransactionDT"].to_numpy()
    )


def test_build_timestamp_uses_configured_reference():
    config = load_config()
    frame = pd.DataFrame({"TransactionDT": [0, 86_400.0]})
    out = build_timestamp(frame, config, "ieee_cis")
    assert out["timestamp"].iloc[0] == pd.Timestamp(config["ieee_reference_date"])
    assert out["timestamp"].iloc[1] == pd.Timestamp(config["ieee_reference_date"]) + pd.Timedelta(days=1)
