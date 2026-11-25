"""Amount-based features for fraud detection (Role 3 — Harsh).

This module is **entirely new** — the reference implementation does not contain
any amount-transform features.

Transformations in this module are row-local (they depend only on the current
row's amount and optionally on a train-set global mean/std).  They therefore
carry no temporal leakage.

For train/test usage:
    * Fit ``AmountFeaturizer`` on the **training fold only**.
    * Apply ``.transform()`` to both train and test without re-fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stateless (no fitting required) amount features
# ---------------------------------------------------------------------------


def log1p_amount(
    frame: pd.DataFrame, *, amount_col: str = "TransactionAmt"
) -> pd.DataFrame:
    """Natural log of (amount + 1) — compresses heavy-tailed distribution.

    ``log1p`` handles zero amounts gracefully (log(1) = 0).
    """
    if amount_col not in frame.columns:
        raise KeyError(f"Amount column {amount_col!r} not found")
    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    return pd.DataFrame(
        {f"{amount_col}_log1p": np.log1p(amount.clip(lower=0))},
        index=frame.index,
    )


def amount_bins(
    frame: pd.DataFrame,
    *,
    amount_col: str = "TransactionAmt",
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Ordinal bin index for transaction amount.

    Default bins match typical retail transaction size categories:
    0–10, 10–50, 50–100, 100–500, 500–1000, 1000+.
    """
    if amount_col not in frame.columns:
        raise KeyError(f"Amount column {amount_col!r} not found")

    _bins = bins or [0, 10, 50, 100, 500, 1_000, np.inf]
    _labels = labels or ["micro", "small", "medium", "large", "xlarge", "xxlarge"]

    amount = pd.to_numeric(frame[amount_col], errors="coerce").clip(lower=0)
    binned = pd.cut(amount, bins=_bins, labels=_labels, right=False)
    return pd.DataFrame(
        {f"{amount_col}_bin": binned},
        index=frame.index,
    )


def amount_bin_codes(
    frame: pd.DataFrame,
    *,
    amount_col: str = "TransactionAmt",
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Integer bin index (0-based) for direct use in tree models."""
    _bins = bins or [0, 10, 50, 100, 500, 1_000, np.inf]
    amount = pd.to_numeric(frame[amount_col], errors="coerce").clip(lower=0)
    codes = pd.cut(amount, bins=_bins, right=False).cat.codes
    return pd.DataFrame(
        {f"{amount_col}_bin_code": codes.astype("Int8")},
        index=frame.index,
    )


def round_amount_flag(
    frame: pd.DataFrame, *, amount_col: str = "TransactionAmt"
) -> pd.DataFrame:
    """Flag transactions whose amount is a round number (integer or x.00).

    Round-amount transactions are over-represented in fraudulent card-testing
    patterns (e.g. £1.00, €100.00 test charges).
    """
    if amount_col not in frame.columns:
        raise KeyError(f"Amount column {amount_col!r} not found")
    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    is_round = (amount % 1 == 0).astype("Int8")
    return pd.DataFrame({f"{amount_col}_is_round": is_round}, index=frame.index)


def cents_component(
    frame: pd.DataFrame, *, amount_col: str = "TransactionAmt"
) -> pd.DataFrame:
    """Fractional (cents) part of the transaction amount (0.00–0.99).

    Psychological pricing (e.g. €19.99) is common in legitimate transactions.
    Fraudsters often use round amounts or unusual cent values.
    """
    if amount_col not in frame.columns:
        raise KeyError(f"Amount column {amount_col!r} not found")
    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    cents = (amount % 1).round(2)
    return pd.DataFrame({f"{amount_col}_cents": cents}, index=frame.index)


# ---------------------------------------------------------------------------
# Stateful features (must be fitted on training data only)
# ---------------------------------------------------------------------------


class AmountFeaturizer:
    """Compute amount features that require train-set statistics.

    Fit on the training fold; transform both train and test.

    Attributes
    ----------
    global_mean_ : float
        Mean transaction amount on the training set.
    global_std_ : float
        Standard deviation of transaction amount on the training set.
    percentile_thresholds_ : dict[int, float]
        Mapping of percentile -> value, e.g. {90: 350.0, 95: 650.0, 99: 1200.0}.
    """

    _PERCENTILES = (75, 90, 95, 99)

    def __init__(self, amount_col: str = "TransactionAmt"):
        self.amount_col = amount_col
        self.global_mean_: float | None = None
        self.global_std_: float | None = None
        self.percentile_thresholds_: dict[int, float] = {}
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "AmountFeaturizer":
        """Compute statistics from the training fold.

        Parameters
        ----------
        train:
            Training-fold DataFrame.  Must contain ``self.amount_col``.
        """
        if self.amount_col not in train.columns:
            raise KeyError(f"Amount column {self.amount_col!r} not found")
        amount = pd.to_numeric(train[self.amount_col], errors="coerce").dropna()
        self.global_mean_ = float(amount.mean())
        self.global_std_ = float(amount.std())
        self.percentile_thresholds_ = {
            p: float(np.percentile(amount, p)) for p in self._PERCENTILES
        }
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply learned statistics to produce new features.

        Parameters
        ----------
        frame:
            Any fold (train or test).

        Returns
        -------
        pd.DataFrame
            New feature columns (original index preserved).

        Features produced
        -----------------
        * ``amount_global_zscore`` — z-score relative to train-set mean/std.
        * ``amount_ratio_to_global_mean`` — amount / train-set mean.
        * ``is_above_p90``, ``is_above_p95``, ``is_above_p99`` — binary flags.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() on the training fold before .transform()")
        if self.amount_col not in frame.columns:
            raise KeyError(f"Amount column {self.amount_col!r} not found")

        amount = pd.to_numeric(frame[amount_col := self.amount_col], errors="coerce")
        output = pd.DataFrame(index=frame.index)

        std = self.global_std_ if self.global_std_ and self.global_std_ > 0 else 1.0
        output["amount_global_zscore"] = (amount - self.global_mean_) / std
        output["amount_ratio_to_global_mean"] = amount / (self.global_mean_ or 1.0)

        for p, threshold in self.percentile_thresholds_.items():
            output[f"is_above_p{p}"] = (amount > threshold).astype("Int8")

        return output

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit on and transform the training fold."""
        return self.fit(train).transform(train)


# ---------------------------------------------------------------------------
# Convenience: build all stateless amount features
# ---------------------------------------------------------------------------


def build_amount_features(
    frame: pd.DataFrame, *, amount_col: str = "TransactionAmt"
) -> pd.DataFrame:
    """Return all stateless amount features concatenated into one DataFrame.

    Safe to call on any fold without risk of leakage (no train-set statistics).

    Features
    --------
    * Log1p transform
    * Bin code (integer)
    * Round-amount flag
    * Cents component
    """
    parts = [
        log1p_amount(frame, amount_col=amount_col),
        amount_bin_codes(frame, amount_col=amount_col),
        round_amount_flag(frame, amount_col=amount_col),
        cents_component(frame, amount_col=amount_col),
    ]
    return pd.concat(parts, axis=1)
