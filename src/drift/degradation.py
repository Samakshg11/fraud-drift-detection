"""REFERENCE IMPLEMENTATION for Role 5 (Drift Analysis) -- safe to replace."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evaluation.metrics import compute_metrics
from src.pipeline.preprocessing import build_preprocessor
from src.pipeline.splits import period_from_month_index


def static_degradation_curve(
    frame: pd.DataFrame,
    *,
    model: object,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    train_months: int,
    fpr_target: float,
    cost_fp: float,
    cost_fn: float,
    time_col: str = "timestamp",
    month_col: str = "month",
    scale_numeric: bool = False,
) -> pd.DataFrame:
    """Train ONCE on the first ``train_months`` and score every later month.

    No refitting happens after the initial training: the resulting
    metric-vs-month curve is the evidence of model degradation. All learned
    preprocessing state also comes from that single training window.
    """
    months = sorted(frame[month_col].unique().tolist())
    if len(months) <= train_months:
        raise ValueError("Need more months than train_months to measure degradation")
    train_month_set = set(months[:train_months])
    train_mask = frame[month_col].isin(train_month_set)
    train = frame.loc[train_mask]
    y_train = train[target_col].to_numpy()
    features = [*numeric_cols, *categorical_cols]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale_numeric)
    matrix_train = preprocessor.fit_transform(train[features])
    fitted = clone(model)
    fitted.fit(matrix_train, y_train)

    rows = []
    for month in months[train_months:]:
        test = frame.loc[frame[month_col] == month]
        y_test = test[target_col].to_numpy()
        if len(np.unique(y_test)) < 2:
            # Metrics are undefined on a single-class month; still record coverage.
            rows.append(
                {
                    "month": month,
                    "month_label": str(period_from_month_index(month)),
                    "n_rows": len(test),
                    "n_fraud": int(y_test.sum()),
                    "pr_auc": float("nan"),
                }
            )
            continue
        matrix_test = preprocessor.transform(test[features])
        scores = fitted.predict_proba(matrix_test)[:, 1]
        row = compute_metrics(
            y_test, scores, fpr_target=fpr_target, cost_fp=cost_fp, cost_fn=cost_fn
        )
        rows.append(
            {
                "month": month,
                "month_label": str(period_from_month_index(month)),
                "n_rows": len(test),
                "n_fraud": int(y_test.sum()),
                **row,
            }
        )
    return pd.DataFrame(rows)
