"""REFERENCE IMPLEMENTATION for Role 6 (Retraining) -- safe to replace.

Every strategy receives a walk-forward ``train_df`` containing only rows from
months strictly before the test month. The strategies decide *when* to refit;
``base.RetrainingStrategy._refit`` guarantees fitting uses train rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift.detectors import ErrorStreamDetector

from .base import RetrainingStrategy


class StaticStrategy(RetrainingStrategy):
    """Train once; never update."""

    def fit(self, train_df: pd.DataFrame) -> None:
        if self.refit_count_ == 0:
            self._refit(train_df)


class PeriodicStrategy(RetrainingStrategy):
    """Retrain every ``period_months`` calendar months."""

    def __init__(self, *, period_months: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if period_months <= 0:
            raise ValueError("period_months must be positive")
        self.period_months = period_months

    def fit(self, train_df: pd.DataFrame) -> None:
        current_month = self._max_month(train_df)
        due = (
            self.refit_count_ == 0
            or current_month - self.last_train_month_ >= self.period_months
        )
        if due:
            self._refit(train_df)


class SlidingWindowStrategy(RetrainingStrategy):
    """Retrain using only the most recent ``window_months`` of data."""

    def __init__(self, *, window_months: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if window_months <= 0:
            raise ValueError("window_months must be positive")
        self.window_months = window_months

    def fit(self, train_df: pd.DataFrame) -> None:
        latest = self._max_month(train_df)
        recent = train_df.loc[train_df[self.month_col] > latest - self.window_months]
        self._refit(recent)


class DriftTriggeredStrategy(RetrainingStrategy):
    """Retrain when a river detector signals drift on the model's error stream."""

    def __init__(
        self,
        *,
        method: str = "adwin",
        detector_params: dict | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.detector_params = detector_params or {}
        self.detector_ = ErrorStreamDetector(method, **self.detector_params)
        self._drift_pending = False

    def update(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        predictions = np.asarray(y_score) >= 0.5
        errors = (predictions != np.asarray(y_true)).astype(int)
        for error in errors:
            if self.detector_.update(int(error)):
                self._drift_pending = True

    def fit(self, train_df: pd.DataFrame) -> None:
        if self.refit_count_ == 0 or self._drift_pending:
            self._refit(train_df)
            # Reset the detector so the new model starts with a clean stream.
            self.detector_ = ErrorStreamDetector(self.method, **self.detector_params)
            self._drift_pending = False


class WeightedStrategy(RetrainingStrategy):
    """Recency-weighted training: exponentially decaying sample weights."""

    def __init__(self, *, half_life_months: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if half_life_months <= 0:
            raise ValueError("half_life_months must be positive")
        self.half_life_months = half_life_months

    def fit(self, train_df: pd.DataFrame) -> None:
        latest = self._max_month(train_df)
        ages = latest - train_df[self.month_col].to_numpy()
        weights = 0.5 ** (ages / self.half_life_months)
        self._refit(train_df, sample_weight=np.asarray(weights, dtype=float))
