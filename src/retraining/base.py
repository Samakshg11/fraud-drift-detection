"""REFERENCE IMPLEMENTATION for Role 6 (Retraining) -- safe to replace.

Common interface shared by every retraining strategy::

    strategy.fit(train_df)          # decided internally whether to (re)train
    scores = strategy.predict_proba(test_df)
    strategy.update(y_true, scores) # optional feedback for drift-triggered
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.pipeline.preprocessing import build_preprocessor


class RetrainingStrategy(ABC):
    """Base class handling preprocessing and the shared fitted state."""

    def __init__(
        self,
        *,
        model: object,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        target_col: str,
        time_col: str = "timestamp",
        month_col: str = "month",
        scale_numeric: bool = False,
        one_hot_min_frequency: int | None = None,
    ) -> None:
        self.model = model
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.features = [*self.numeric_cols, *self.categorical_cols]
        self.target_col = target_col
        self.time_col = time_col
        self.month_col = month_col
        self.scale_numeric = scale_numeric
        self.one_hot_min_frequency = one_hot_min_frequency

        self.model_: object | None = None
        self.preprocessor_: object | None = None
        self.refit_count_ = 0
        self.last_train_rows_ = 0
        self.last_train_month_: int | None = None

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit or refit using only ``train_df`` (already temporally earlier)."""

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.preprocessor_ is None:
            raise RuntimeError("Strategy must be fitted before predict_proba")
        matrix = self.preprocessor_.transform(test_df[self.features])
        return np.asarray(self.model_.predict_proba(matrix)[:, 1], dtype=float)

    def update(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Optional label feedback hook (used by drift-triggered strategies)."""

    def _max_month(self, frame: pd.DataFrame) -> int:
        return int(frame[self.month_col].max())

    def _refit(
        self, train_df: pd.DataFrame, *, sample_weight: np.ndarray | None = None
    ) -> None:
        """Fit preprocessing + model on train_df only. The sole fitting path."""
        if train_df.empty:
            raise ValueError("Cannot fit on an empty training frame")
        y = train_df[self.target_col].to_numpy()
        self.preprocessor_ = build_preprocessor(
            self.numeric_cols,
            self.categorical_cols,
            scale_numeric=self.scale_numeric,
            min_frequency=self.one_hot_min_frequency,
        )
        matrix = self.preprocessor_.fit_transform(train_df[self.features])
        self.model_ = clone(self.model)
        fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        self.model_.fit(matrix, y, **fit_kwargs)
        self.refit_count_ += 1
        self.last_train_rows_ = len(train_df)
        self.last_train_month_ = self._max_month(train_df)
