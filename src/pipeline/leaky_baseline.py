"""Side-by-side leaky and leakage-free baselines.

.. warning::
   ``run_leaky_baseline`` is **deliberately invalid**. It uses a random
   stratified split, fits target/frequency encoders and preprocessing on *all*
   rows (including the test rows), and can add aggregates computed over the
   full timeline. It exists only to quantify how much these mistakes inflate
   reported metrics. Never use its numbers as research results.

``run_leakage_free_baseline`` is the correct counterpart: walk-forward month
splits, past-only encodings on training rows, frozen train-period mappings on
test rows, and all learned state fitted on each training fold only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .encoders import FrequencyEncoder, TimeAwareTargetEncoder
from .preprocessing import build_preprocessor
from .splits import walk_forward_splits

LEAKY_WARNING = (
    "DELIBERATELY LEAKY: random split + full-data preprocessing/encoders. "
    "For inflation measurement only; not a valid evaluation."
)


@dataclass
class BaselineResult:
    """Predictions and metadata from one baseline run."""

    label: str
    y_true: np.ndarray
    y_score: np.ndarray
    fold_summaries: list[dict] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(len(self.y_true))


def leaky_random_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.Index, pd.Index]:
    """Invalid: random stratified split that mixes past and future rows."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
        shuffle=True,
    )
    return train_idx, test_idx


def add_global_aggregates(
    df: pd.DataFrame, *, group_col: str, value_col: str
) -> tuple[pd.DataFrame, list[str]]:
    """Invalid: aggregates computed over the whole timeline, including future.

    Returns the frame with two added numeric columns and their names, so the
    caller can include them in the numeric feature list.
    """
    for col in (group_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found")
    out = df.copy()
    mean_col = f"{group_col}_global_mean"
    count_col = f"{group_col}_global_count"
    grouped = out.groupby(group_col, sort=False)[value_col]
    out[mean_col] = grouped.transform("mean")
    out[count_col] = grouped.transform("count")
    return out, [mean_col, count_col]


def _feature_lists(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_encoded_cols: Sequence[str],
    frequency_encoded_cols: Sequence[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (input_cols, model_numeric, model_categorical, model_cols)."""
    encoded = set(target_encoded_cols) | set(frequency_encoded_cols)
    overlap_numeric = encoded & set(numeric_cols)
    if overlap_numeric:
        raise ValueError(f"Encoded columns must be categorical, not numeric: {overlap_numeric}")
    input_cols = list(dict.fromkeys([*numeric_cols, *categorical_cols]))
    model_numeric = [*numeric_cols, *target_encoded_cols, *frequency_encoded_cols]
    model_categorical = [c for c in categorical_cols if c not in encoded]
    model_cols = [*model_numeric, *model_categorical]
    return input_cols, model_numeric, model_categorical, model_cols


def run_leaky_baseline(
    df: pd.DataFrame,
    *,
    target_col: str,
    time_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    model: object,
    seed: int,
    test_size: float = 0.2,
    target_encoded_cols: Sequence[str] = (),
    frequency_encoded_cols: Sequence[str] = (),
    group_col: str | None = None,
    value_col: str | None = None,
    scale_numeric: bool = False,
) -> BaselineResult:
    """Run the deliberately invalid baseline (for inflation measurement only).

    Leakage sources, on purpose:
    * random stratified split instead of chronological;
    * target/frequency encoders fitted on the *full* frame and applied to it,
      so a row's own label contributes to its feature;
    * preprocessor fitted on the *full* frame;
    * optional group aggregates computed over future rows.
    """
    if (group_col is None) != (value_col is None):
        raise ValueError("group_col and value_col must be supplied together")

    working = df
    numeric = list(numeric_cols)
    if group_col is not None:
        working, added = add_global_aggregates(
            working, group_col=group_col, value_col=value_col  # type: ignore[arg-type]
        )
        numeric = [*numeric, *added]

    y = working[target_col].to_numpy()
    train_idx, test_idx = leaky_random_split(
        working, target_col=target_col, test_size=test_size, seed=seed
    )
    # SciPy sparse matrices index by position, not by DataFrame index label.
    positions = pd.Series(np.arange(len(working)), index=working.index)
    train_pos = positions.loc[train_idx].to_numpy()
    test_pos = positions.loc[test_idx].to_numpy()

    input_cols, model_numeric, model_categorical, model_cols = _feature_lists(
        numeric, categorical_cols, target_encoded_cols, frequency_encoded_cols
    )
    input_cols = list(dict.fromkeys([*input_cols, time_col]))

    # Fit encoders on ALL rows (the leak) and transform ALL rows.
    X_all = working[input_cols].copy()
    if target_encoded_cols:
        encoder = TimeAwareTargetEncoder(
            list(target_encoded_cols), time_col=time_col
        ).fit(X_all, y)
        X_all = encoder.transform(X_all)
    if frequency_encoded_cols:
        freq = FrequencyEncoder(list(frequency_encoded_cols), time_col=time_col).fit(X_all)
        X_all = freq.transform(X_all)

    preprocessor = build_preprocessor(
        model_numeric, model_categorical, scale_numeric=scale_numeric
    )
    matrix = preprocessor.fit_transform(X_all[model_cols])  # leak: fitted on test too

    fitted = clone(model)
    fitted.fit(matrix[train_pos], y[train_pos])
    scores = fitted.predict_proba(matrix[test_pos])[:, 1]

    return BaselineResult(
        label="leaky",
        y_true=np.asarray(y[test_pos], dtype=int),
        y_score=np.asarray(scores, dtype=float),
        fold_summaries=[
            {
                "fold": 0,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "note": LEAKY_WARNING,
            }
        ],
    )


def run_leakage_free_baseline(
    df: pd.DataFrame,
    *,
    target_col: str,
    time_col: str,
    month_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    model: object,
    train_months: int,
    test_months: int,
    gap_days: int = 0,
    scheme: str = "expanding",
    sliding_window_months: int | None = None,
    fpr_target: float = 0.01,
    target_encoded_cols: Sequence[str] = (),
    frequency_encoded_cols: Sequence[str] = (),
    scale_numeric: bool = False,
) -> BaselineResult:
    """Run the leakage-free walk-forward baseline.

    Within every fold:

    * split by calendar month with an embargo;
    * ``fit_transform`` the encoders on train rows only (expanding, past-only);
    * ``transform`` test rows with the frozen train-period mapping;
    * fit imputer/scaler/one-hot on train rows only;
    * resampling (if any) would happen here, on train rows only.

    ``fpr_target`` is unused here but accepted so callers can pass a uniform
    keyword set to both baselines; metric computation belongs to the caller.
    """
    del fpr_target  # accepted for a uniform interface; metrics are external
    input_cols, model_numeric, model_categorical, model_cols = _feature_lists(
        numeric_cols, categorical_cols, target_encoded_cols, frequency_encoded_cols
    )
    input_cols = list(dict.fromkeys([*input_cols, time_col]))

    y_true_parts: list[np.ndarray] = []
    y_score_parts: list[np.ndarray] = []
    summaries: list[dict] = []

    splits = walk_forward_splits(
        df,
        time_col=time_col,
        month_col=month_col,
        train_months=train_months,
        test_months=test_months,
        gap_days=gap_days,
        scheme=scheme,
        sliding_window_months=sliding_window_months,
    )
    for fold, split in enumerate(splits):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]
        y_train = train[target_col].to_numpy()
        y_test = test[target_col].to_numpy()
        X_train = train[input_cols].copy()
        X_test = test[input_cols].copy()

        # Encoders: past-only values for train, frozen mapping for test.
        if target_encoded_cols:
            encoder = TimeAwareTargetEncoder(
                list(target_encoded_cols), time_col=time_col
            )
            X_train = encoder.fit_transform(X_train, y_train)
            X_test = encoder.transform(X_test)
        if frequency_encoded_cols:
            freq = FrequencyEncoder(list(frequency_encoded_cols), time_col=time_col)
            X_train = freq.fit_transform(X_train)
            X_test = freq.transform(X_test)

        preprocessor = build_preprocessor(
            model_numeric, model_categorical, scale_numeric=scale_numeric
        )
        X_train_matrix = preprocessor.fit_transform(X_train[model_cols])
        X_test_matrix = preprocessor.transform(X_test[model_cols])

        fitted = clone(model)
        fitted.fit(X_train_matrix, y_train)
        y_score_parts.append(fitted.predict_proba(X_test_matrix)[:, 1])
        y_true_parts.append(np.asarray(y_test, dtype=int))
        summaries.append(
            {
                "fold": fold,
                "train_rows": len(split.train_idx),
                "test_rows": len(split.test_idx),
                "train_end": split.train_end,
                "test_start": split.test_start,
            }
        )

    if not y_true_parts:
        raise ValueError("Walk-forward produced no folds; check split configuration")

    return BaselineResult(
        label="leakage_free",
        y_true=np.concatenate(y_true_parts),
        y_score=np.concatenate(y_score_parts),
        fold_summaries=summaries,
    )
