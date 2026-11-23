"""REFERENCE IMPLEMENTATION for Role 4 (Modeling) -- safe to replace.

Primary metrics for imbalanced fraud detection. Accuracy is deliberately not
included. Every metric table must also report the fraud base rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


@dataclass(frozen=True)
class RecallAtFPR:
    """Operating point satisfying an FPR ceiling."""

    recall: float
    achieved_fpr: float
    threshold: float


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, fpr_target: float
) -> RecallAtFPR:
    """Best recall among thresholds whose false-positive rate <= ``fpr_target``.

    ``drop_intermediate=False`` keeps all thresholds; with a tiny fraud count
    the default ROC pruning can otherwise remove every low-FPR operating point.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return RecallAtFPR(float("nan"), float("nan"), float("nan"))
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= fpr_target)
    if len(eligible) == 0:
        return RecallAtFPR(0.0, 0.0, float(np.max(y_score) + 1.0))
    best = int(eligible[np.argmax(tpr[eligible])])
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):  # roc_curve's first point is at +inf
        threshold = float(np.max(y_score) + 1.0)
    return RecallAtFPR(float(tpr[best]), float(fpr[best]), threshold)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    fpr_target: float,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, float | int | str]:
    """Compute PR-AUC, recall@FPR, F1/precision/recall, cost and base rate."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    point = recall_at_fpr(y_true, y_score, fpr_target)

    if np.isfinite(point.threshold):
        y_pred = (y_score >= point.threshold).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        false_pos = int(((y_pred == 1) & (y_true == 0)).sum())
        false_neg = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = false_pos * cost_fp + false_neg * cost_fn
    else:
        precision = f1 = recall = cost = float("nan")

    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_fpr": point.recall,
        "achieved_fpr": point.achieved_fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost_at_fpr": float(cost),
        "fraud_base_rate": float(y_true.mean()),
        "n_rows": int(len(y_true)),
    }


def compute_metrics_from_config(
    y_true: np.ndarray, y_score: np.ndarray, config: dict, *, setup: str | None = None
) -> dict:
    """Convenience wrapper that reads thresholds/costs from config."""
    row = compute_metrics(
        y_true,
        y_score,
        fpr_target=config["fpr_target"],
        cost_fp=config["cost_fp"],
        cost_fn=config["cost_fn"],
    )
    if setup is not None:
        row = {"setup": setup, **row}
    return row


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Order columns consistently for CSV output and plotting."""
    table = pd.DataFrame(rows)
    preferred = [
        "setup",
        "pr_auc",
        "recall_at_fpr",
        "achieved_fpr",
        "precision",
        "recall",
        "f1",
        "cost_at_fpr",
        "fraud_base_rate",
        "n_rows",
    ]
    return table[[c for c in preferred if c in table.columns]]
