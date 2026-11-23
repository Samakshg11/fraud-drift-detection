"""Tests for the Role 4 reference metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import compute_metrics, recall_at_fpr


def test_recall_at_fpr_selects_best_recall_under_ceiling():
    y = np.array([0, 0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.9])
    point = recall_at_fpr(y, score, fpr_target=0.3)
    assert point.achieved_fpr <= 0.3
    assert point.recall == pytest.approx(1.0)
    assert 0 < point.threshold <= 0.6


def test_recall_at_fpr_handles_no_eligible_threshold():
    y = np.array([0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6])  # fraud scored lowest
    point = recall_at_fpr(y, score, fpr_target=0.01)
    assert point.recall == 0.0
    assert np.isfinite(point.threshold)


def test_metrics_include_base_rate_and_cost():
    y = np.array([0] * 99 + [1])
    score = np.linspace(0, 1, 100)
    row = compute_metrics(y, score, fpr_target=0.01, cost_fp=1.0, cost_fn=10.0)
    assert row["fraud_base_rate"] == pytest.approx(0.01)
    assert row["pr_auc"] == pytest.approx(average_precision_score(y, score))
    assert row["n_rows"] == 100
    assert row["cost_at_fpr"] >= 0
