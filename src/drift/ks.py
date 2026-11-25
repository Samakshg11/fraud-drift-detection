"""REFERENCE IMPLEMENTATION for Role 5 (Drift Analysis) -- safe to replace."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns ``(statistic, p_value)``. A small p-value means the two samples are
    unlikely to come from the same distribution (i.e. data drift).
    """
    ref = np.asarray(reference, dtype=float).flatten()
    cur = np.asarray(current, dtype=float).flatten()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan"), float("nan")
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)
