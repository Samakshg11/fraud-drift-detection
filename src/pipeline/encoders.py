"""Time-aware categorical encoders (sklearn-compatible).

Leakage rule implemented here
-----------------------------
A row's encoding may only use information from rows that occurred *strictly
earlier in time*. Rows sharing the exact same timestamp never influence each
other (they are aggregated into one time bucket, then subtracted out).

There are two distinct modes, matching how sklearn pipelines are used:

* ``fit_transform(train_X, y)`` returns **expanding, past-only** encodings for
  the training rows, so the model never sees its own target echoed back.
* ``fit(train_X, y)`` followed by ``transform(test_X)`` applies a **frozen
  mapping learned from the entire training period** to future rows. Test/time-
  future rows never contribute to any statistic.

This distinction is deliberate: at scoring time we do not know the labels of
earlier test rows (fraud labels arrive late), so expanding updates on test data
would itself be leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_MISSING_TOKEN = "__MISSING__"


def _categories(series: pd.Series) -> pd.Series:
    """Normalise a categorical column to strings, including a NaN sentinel."""
    return series.fillna(_MISSING_TOKEN).astype(str)


def _past_only_cumulatives(
    categories: pd.Series,
    times: pd.Series,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row (past category sum, past category count, past global count).

    All values exclude the current row and every row with the same timestamp.
    Results are returned in the original row order.
    """
    frame = pd.DataFrame(
        {
            "_row": np.arange(len(categories)),
            "_cat": _categories(categories).to_numpy(),
            "_time": pd.Series(times).to_numpy(),
            "_y": np.asarray(target, dtype=float),
        }
    )
    if frame["_time"].isna().any():
        raise ValueError("time column contains NaNs; cannot compute past-only encodings")

    by_cat_time = (
        frame.groupby(["_cat", "_time"], sort=True)["_y"]
        .agg(["sum", "count"])
        .reset_index()
    )
    by_cat_time["cum_sum"] = (
        by_cat_time.groupby("_cat", sort=False)["sum"].cumsum() - by_cat_time["sum"]
    )
    by_cat_time["cum_count"] = (
        by_cat_time.groupby("_cat", sort=False)["count"].cumsum()
        - by_cat_time["count"]
    )

    by_time = frame.groupby("_time", sort=True)["_y"].agg(["sum", "count"]).reset_index()
    by_time["past_global_sum"] = by_time["sum"].cumsum() - by_time["sum"]
    by_time["past_global_count"] = by_time["count"].cumsum() - by_time["count"]

    joined = frame.merge(
        by_cat_time[["_cat", "_time", "cum_sum", "cum_count"]],
        on=["_cat", "_time"],
        how="left",
        validate="many_to_one",
    ).merge(
        by_time[["_time", "past_global_sum", "past_global_count"]],
        on="_time",
        how="left",
        validate="many_to_one",
    )
    joined = joined.sort_values("_row", kind="stable")
    return (
        joined["cum_sum"].to_numpy(dtype=float),
        joined["cum_count"].to_numpy(dtype=float),
        joined["past_global_count"].to_numpy(dtype=float),
    )


class TimeAwareTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding using only strictly earlier rows.

    Parameters
    ----------
    columns:
        Categorical column names to encode. Each is replaced in place by a
        float column in the returned DataFrame.
    time_col:
        Column used to order rows. If ``None``, existing row order is treated
        as time order (the input must already be time-sorted).
    smoothing:
        Pseudo-count strength of the smoothing. Higher = more shrinkage toward
        the prior.
    prior:
        Fallback prior used when there is no past data at all.
    handle_unknown:
        ``"use_prior"`` (default) or ``"error"`` for categories unseen in fit.

    Notes
    -----
    ``fit_transform`` is *not* equivalent to ``fit`` then ``transform`` on the
    same frame: the former returns expanding values (correct for training), the
    latter returns the frozen mapping (correct for future/test rows).
    """

    def __init__(
        self,
        columns: list[str],
        *,
        time_col: str | None = None,
        smoothing: float = 10.0,
        prior: float = 0.0,
        handle_unknown: str = "use_prior",
    ) -> None:
        self.columns = list(columns)
        self.time_col = time_col
        self.smoothing = smoothing
        self.prior = prior
        self.handle_unknown = handle_unknown

    def _check(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TimeAwareTargetEncoder expects a pandas DataFrame")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        if self.time_col is not None and self.time_col not in X.columns:
            raise KeyError(f"time_col {self.time_col!r} not found")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if self.handle_unknown not in ("use_prior", "error"):
            raise ValueError("handle_unknown must be 'use_prior' or 'error'")

    def _times(self, X: pd.DataFrame) -> pd.Series:
        if self.time_col is None:
            return pd.Series(np.arange(len(X)), index=X.index)
        return pd.to_datetime(X[self.time_col])

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "TimeAwareTargetEncoder":
        """Learn the frozen per-category mapping from the training period only."""
        self._check(X)
        target = np.asarray(y, dtype=float)
        if len(target) != len(X):
            raise ValueError("X and y have different lengths")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("Target must be binary 0/1")
        self.prior_ = float(target.mean()) if len(target) else float(self.prior)
        self.mapping_: dict[str, pd.Series] = {}

        work = pd.DataFrame(
            {"_cat": _categories(X[col]).to_numpy() for col in self.columns}
        )
        work["_y"] = target
        for col in self.columns:
            stats = work.groupby("_cat", sort=False)["_y"].agg(["sum", "count"])
            self.mapping_[col] = (
                stats["sum"] + self.smoothing * self.prior_
            ) / (stats["count"] + self.smoothing)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen mapping to future rows (never refits)."""
        self._check(X)
        if not hasattr(self, "mapping_"):
            raise RuntimeError("Encoder must be fitted before transform")
        out = X.copy()
        for col in self.columns:
            mapped = _categories(out[col]).map(self.mapping_[col])
            if mapped.isna().any() and self.handle_unknown == "error":
                raise ValueError(f"Unseen category in column {col!r} during transform")
            out[col] = mapped.fillna(self.prior_).astype(float)
        return out

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: object
    ) -> pd.DataFrame:
        """Fit the frozen mapping and return past-only training encodings."""
        self.fit(X, y, **fit_params)
        target = np.asarray(y, dtype=float)
        times = self._times(X)

        # Shrink toward the *past* global mean, never toward the full-data mean.
        past_global = pd.DataFrame({"y": target, "t": times.to_numpy()})
        per_time = past_global.groupby("t", sort=True)["y"].agg(["sum", "count"])
        per_time["past_sum"] = per_time["sum"].cumsum() - per_time["sum"]
        per_time["past_count"] = per_time["count"].cumsum() - per_time["count"]
        prior_row = (
            times.map(per_time["past_sum"] / per_time["past_count"])
            .fillna(float(self.prior))
            .to_numpy(dtype=float)
        )

        out = X.copy()
        for col in self.columns:
            past_sum, past_count, _ = _past_only_cumulatives(X[col], times, target)
            out[col] = (past_sum + self.smoothing * prior_row) / (
                past_count + self.smoothing
            )
        return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Past-only frequency encoding with a frozen mapping for future rows.

    For training rows the value is the number of strictly earlier rows sharing
    the category (or that count divided by the number of earlier rows overall
    when ``normalize=True``). Future rows use the total training-period count.

    Parameters
    ----------
    columns:
        Categorical columns to encode (replaced in place).
    time_col:
        Ordering column; ``None`` means existing row order is time order.
    normalize:
        If True, return past share instead of raw past count.
    handle_unknown:
        ``"use_zero"`` (default) or ``"error"``.
    """

    def __init__(
        self,
        columns: list[str],
        *,
        time_col: str | None = None,
        normalize: bool = False,
        handle_unknown: str = "use_zero",
    ) -> None:
        self.columns = list(columns)
        self.time_col = time_col
        self.normalize = normalize
        self.handle_unknown = handle_unknown

    def _check(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder expects a pandas DataFrame")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        if self.time_col is not None and self.time_col not in X.columns:
            raise KeyError(f"time_col {self.time_col!r} not found")
        if self.handle_unknown not in ("use_zero", "error"):
            raise ValueError("handle_unknown must be 'use_zero' or 'error'")

    def _times(self, X: pd.DataFrame) -> pd.Series:
        if self.time_col is None:
            return pd.Series(np.arange(len(X)), index=X.index)
        return pd.to_datetime(X[self.time_col])

    def fit(self, X: pd.DataFrame, y: None = None) -> "FrequencyEncoder":
        """Learn frozen category counts from the training period only."""
        self._check(X)
        self.counts_: dict[str, pd.Series] = {}
        for col in self.columns:
            self.counts_[col] = _categories(X[col]).value_counts()
        self.total_ = float(len(X))
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training counts to future rows (never refits)."""
        self._check(X)
        if not hasattr(self, "counts_"):
            raise RuntimeError("Encoder must be fitted before transform")
        out = X.copy()
        for col in self.columns:
            counts = _categories(out[col]).map(self.counts_[col])
            if counts.isna().any() and self.handle_unknown == "error":
                raise ValueError(f"Unseen category in column {col!r} during transform")
            values = counts.fillna(0.0).astype(float)
            if self.normalize and self.total_ > 0:
                values = values / self.total_
            out[col] = values
        return out

    def fit_transform(
        self, X: pd.DataFrame, y: None = None, **fit_params: object
    ) -> pd.DataFrame:
        """Fit frozen counts and return past-only frequencies for training rows."""
        self.fit(X, y, **fit_params)
        times = self._times(X)
        dummy_target = np.ones(len(X), dtype=float)
        out = X.copy()
        for col in self.columns:
            _, past_count, past_global_count = _past_only_cumulatives(
                X[col], times, dummy_target
            )
            if self.normalize:
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.divide(
                        past_count,
                        past_global_count,
                        out=np.zeros_like(past_count),
                        where=past_global_count > 0,
                    )
            else:
                values = past_count
            out[col] = values
        return out
