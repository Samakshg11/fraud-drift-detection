"""Tests for the Role 5 reference drift implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.drift.degradation import static_degradation_curve
from src.drift.detectors import detect_drift_indices
from src.drift.ks import ks_statistic
from src.drift.psi import psi, psi_table


def test_psi_is_zero_for_identical_and_high_for_shifted_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 1000)
    assert psi(reference, reference) == 0.0
    assert psi(reference, rng.normal(3, 1, 1000)) > 0.2


def test_psi_table_skips_constant_columns():
    frame = pd.DataFrame({"amount": np.arange(100), "constant": np.ones(100)})
    table = psi_table(frame, frame * 2, ["amount", "constant"])
    assert table.loc[table["column"] == "constant", "psi"].isna().all()
    assert np.isfinite(table.loc[table["column"] == "amount", "psi"]).all()


def test_ks_detects_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 500)
    _, p_same = ks_statistic(reference, rng.normal(0, 1, 500))
    stat, p_shift = ks_statistic(reference, rng.normal(4, 1, 500))
    assert p_same > 0.05
    assert stat > 0.5 and p_shift < 0.05


def test_all_detectors_flag_an_error_rate_change():
    errors = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    configurations = {
        "adwin": {"delta": 0.05, "min_window_length": 5, "grace_period": 5},
        "page_hinkley": {"min_instances": 5, "threshold": 20.0, "delta": 0.01, "alpha": 0.99},
        "ddm": {"warm_start": 10, "warning_threshold": 1.0, "drift_threshold": 2.0},
    }
    for method, params in configurations.items():
        indices = detect_drift_indices(errors, method, **params)
        assert indices, f"{method} failed to flag a complete error-rate change"


def test_static_degradation_curve_trains_once(make_synthetic):
    frame = make_synthetic(months=12, n_per_month=100)
    config = load_config()
    curve = static_degradation_curve(
        frame,
        model=LogisticRegression(max_iter=500, random_state=0),
        numeric_cols=["TransactionAmt", "card1", "addr1"],
        categorical_cols=["ProductCD", "P_emaildomain"],
        target_col="isFraud",
        train_months=6,
        fpr_target=config["fpr_target"],
        cost_fp=config["cost_fp"],
        cost_fn=config["cost_fn"],
        scale_numeric=True,
    )
    assert len(curve) == 6
    assert curve["month"].is_monotonic_increasing
    assert (curve["n_rows"] > 0).all()
