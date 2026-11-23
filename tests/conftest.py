"""Shared synthetic-data fixtures that mimic the two real dataset schemas.

No real data is stored in the repo, so every test uses small generated frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

IEEE_REFERENCE = "2017-12-01"


@pytest.fixture
def make_synthetic():
    """Return a factory for time-sorted synthetic transaction frames.

    The generator adds a fraud-rate drift over time and a categorical signal,
    so leakage-sensitive tests have something realistic to detect.
    """

    def _make(
        *,
        schema: str = "ieee",
        n_per_month: int = 100,
        months: int = 12,
        seed: int = 7,
        start: str = "2019-01-01",
        with_timestamps: bool = True,
    ) -> pd.DataFrame:
        if schema not in ("ieee", "sparkov"):
            raise ValueError("schema must be 'ieee' or 'sparkov'")
        rng = np.random.default_rng(seed)
        month_offsets = np.repeat(np.arange(months), n_per_month)
        base = pd.Timestamp(start).to_period("M")
        # Spread rows over the first 28 days of each month so embargo tests are
        # meaningful (a 10-day gap must leave later rows in the test block).
        timestamps = []
        for position, offset in enumerate(month_offsets):
            within_month = position % n_per_month
            timestamps.append(
                (base + int(offset)).to_timestamp()
                + pd.Timedelta(days=within_month % 28, hours=(within_month * 7) % 24)
            )
        timestamps = pd.Series(pd.to_datetime(timestamps))
        n = len(timestamps)

        amount = np.round(rng.lognormal(mean=3.0, sigma=0.6, size=n), 2)
        n_cat = 4
        cat = rng.choice([f"c{i}" for i in range(n_cat)], size=n)
        fraud_rate = 0.005 + 0.02 * (month_offsets / max(months - 1, 1))
        score = (
            np.log(fraud_rate / (1 - fraud_rate))
            + 0.9 * (cat == "c0")
            + 0.4 * (amount > np.quantile(amount, 0.9))
        )
        prob = 1.0 / (1.0 + np.exp(-score))
        target = rng.binomial(1, prob).astype(int)

        frame = pd.DataFrame({"timestamp": timestamps})
        frame["month"] = (timestamps.dt.year * 12 + timestamps.dt.month - 1).to_numpy()

        if schema == "ieee":
            frame["TransactionID"] = np.arange(n)
            frame["TransactionDT"] = (
                timestamps - pd.Timestamp(IEEE_REFERENCE)
            ).dt.total_seconds().to_numpy()
            frame["TransactionAmt"] = amount
            frame["ProductCD"] = cat
            frame["card1"] = rng.integers(1000, 1100, size=n).astype(float)
            frame["addr1"] = rng.normal(200, 50, size=n)
            frame.loc[rng.choice(n, size=n // 20, replace=False), "addr1"] = np.nan
            frame["P_emaildomain"] = rng.choice(["gmail.com", "yahoo.com", None], size=n)
            frame["isFraud"] = target
        else:
            frame["trans_num"] = [f"txn-{i}" for i in range(n)]
            frame["trans_date_trans_time"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
            frame["amt"] = amount
            frame["cc_num"] = rng.integers(10**15, 10**16, size=n)
            frame["merchant"] = rng.choice(["merchant_a", "merchant_b", "merchant_c"], size=n)
            frame["category"] = cat
            frame["state"] = rng.choice(["CA", "NY", "TX"], size=n)
            frame["is_fraud"] = target

        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if not with_timestamps:
            frame = frame.drop(columns=["timestamp", "month"])
        return frame

    return _make
