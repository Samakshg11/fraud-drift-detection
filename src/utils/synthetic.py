"""Shared synthetic demo generator.

Used by scripts for smoke tests when the real datasets are not present.
Numbers obtained from this frame are NEVER research results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.splits import add_month_index


def make_synthetic_demo(seed: int, *, n_per_month: int = 120, months: int = 12) -> pd.DataFrame:
    """Generate a time-sorted frame mimicking IEEE-CIS column names."""
    rng = np.random.default_rng(seed)
    month_offsets = np.repeat(np.arange(months), n_per_month)
    base = pd.Timestamp("2019-01-01").to_period("M")
    timestamps = []
    for position, offset in enumerate(month_offsets):
        within = position % n_per_month
        timestamps.append(
            (base + int(offset)).to_timestamp()
            + pd.Timedelta(days=within % 28, hours=(within * 7) % 24)
        )
    times = pd.Series(pd.to_datetime(timestamps))
    n = len(times)
    amount = np.round(rng.lognormal(3.0, 0.6, n), 2)
    merchant = rng.choice(["m_a", "m_b", "m_c", "m_d"], n)
    rate = 0.005 + 0.02 * (month_offsets / (months - 1))
    logit = np.log(rate / (1 - rate)) + 0.9 * (merchant == "m_a")
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    frame = pd.DataFrame(
        {
            "timestamp": times,
            "TransactionID": np.arange(n),
            "TransactionAmt": amount,
            "card1": rng.integers(1000, 1100, n).astype(float),
            "ProductCD": merchant,
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "isFraud": y,
        }
    )
    return add_month_index(frame, time_col="timestamp").sort_values(
        "timestamp", kind="stable"
    )
