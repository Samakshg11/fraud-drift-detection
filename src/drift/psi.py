"""REFERENCE IMPLEMENTATION for Role 5 (Drift Analysis) -- safe to replace."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Bins are quantiles of the REFERENCE sample, so a PSI value always answers
    "how far has the current distribution moved from the training baseline?".

    Typical reading: <0.1 stable, 0.1-0.2 moderate shift, >0.2 large shift.
    """
    if buckets < 2:
        raise ValueError("buckets must be at least 2")
    e = np.asarray(expected, dtype=float).flatten()
    a = np.asarray(actual, dtype=float).flatten()
    if len(e) == 0 or len(a) == 0:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    edges = np.unique(np.quantile(e, quantiles))
    edges = np.concatenate(([-np.inf], edges, [np.inf]))

    expected_pct = np.histogram(e, bins=edges)[0].astype(float)
    actual_pct = np.histogram(a, bins=edges)[0].astype(float)
    expected_pct = np.clip(expected_pct / expected_pct.sum(), _EPS, None)
    actual_pct = np.clip(actual_pct / actual_pct.sum(), _EPS, None)
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_table(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], *, buckets: int = 10
) -> pd.DataFrame:
    """PSI per numeric column, ignoring columns with no variance in reference."""
    rows = []
    for column in columns:
        ref = reference[column].dropna()
        cur = current[column].dropna()
        value = float("nan") if ref.nunique() < 2 else psi(ref, cur, buckets=buckets)
        rows.append({"column": column, "psi": value})
    return pd.DataFrame(rows)
