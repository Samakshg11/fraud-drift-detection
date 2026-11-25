"""Tests proving preprocessing state is learned from training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.preprocessing import build_preprocessor, infer_feature_columns


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, np.nan, 4.0, 1000.0, 2000.0],
            "product": ["a", "b", "a", "b", "a", "z"],
        }
    )


def test_fit_on_train_only_ignores_extreme_test_values():
    frame = _frame()
    train, test = frame.iloc[:4], frame.iloc[4:]

    train_only = build_preprocessor(
        ["amount"], ["product"], scale_numeric=True, one_hot_sparse=False
    ).fit(train)
    with_test = build_preprocessor(
        ["amount"], ["product"], scale_numeric=True, one_hot_sparse=False
    ).fit(frame)

    scaler_train = train_only.named_transformers_["num"].named_steps["scaler"]
    scaler_full = with_test.named_transformers_["num"].named_steps["scaler"]
    assert not np.allclose(scaler_train.mean_, scaler_full.mean_)

    out_train_only = np.asarray(train_only.transform(test))
    out_full = np.asarray(with_test.transform(test))
    # Different widths already prove the test rows changed the fitted vocabulary.
    assert out_train_only.shape != out_full.shape or not np.allclose(
        out_train_only, out_full
    )

    # Transforming test rows must not mutate fitted statistics.
    before = scaler_train.mean_.copy()
    train_only.transform(test)
    np.testing.assert_array_equal(before, scaler_train.mean_)


def test_imputation_and_unknown_categories_are_handled():
    frame = _frame()
    train, test = frame.iloc[:4], frame.iloc[4:]
    prep = build_preprocessor(
        ["amount"], ["product"], one_hot_sparse=False
    ).fit(train)
    out = prep.transform(test)
    assert not np.isnan(np.asarray(out, dtype=float)).any()
    # Product "z" was unseen in train: its one-hot block must be all zeros.
    names = prep.get_feature_names_out()
    product_columns = [i for i, name in enumerate(names) if "product_" in name]
    # Unknown categories get no column at all under handle_unknown="ignore".
    assert not any(name.endswith("product_z") for name in names)
    assert len(product_columns) > 0
    assert np.asarray(out)[-1, product_columns].sum() == 0  # last row has "z"


def test_missing_indicator_and_scaling_toggle():
    frame = _frame()
    train = frame.iloc[:4]
    prep_raw = build_preprocessor(["amount"], [], add_missing_indicator=True).fit(train)
    assert prep_raw.transform(train).shape[1] == 2  # value + missing flag

    prep_scaled = build_preprocessor(["amount"], [], scale_numeric=True).fit(train)
    scaled = np.asarray(prep_scaled.transform(train))
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-6)


def test_infer_feature_columns_excludes_metadata():
    frame = pd.DataFrame(
        {
            "isFraud": [0, 1],
            "timestamp": pd.to_datetime(["2019-01-01", "2019-01-02"]),
            "month": [1, 1],
            "TransactionID": [1, 2],
            "TransactionAmt": [1.0, 2.0],
            "ProductCD": ["a", "b"],
        }
    )
    numeric, categorical = infer_feature_columns(
        frame, exclude=["isFraud", "timestamp", "month", "TransactionID"]
    )
    assert numeric == ["TransactionAmt"]
    assert categorical == ["ProductCD"]
