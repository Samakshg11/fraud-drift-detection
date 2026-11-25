"""REFERENCE IMPLEMENTATION for Role 1 (Data Engineer) -- safe to replace.

Only deterministic, row-local transformations happen here. Learned imputation,
scaling and encoding belong to ``src/pipeline/preprocessing.py`` and must be
fitted on training folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.splits import add_month_index, derive_datetime


def downcast_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer/float columns while preserving timestamp precision.

    Float32 has ~7 significant digits, so a seconds offset of 6e7 would lose
    whole seconds. Columns larger than 2**24 stay float64.
    """
    out = frame.copy(deep=False)
    float_limit = 2**24
    for col in out.select_dtypes(include=["integer"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["floating"]).columns:
        series = out[col]
        finite = series.dropna()
        if len(finite) and float(finite.abs().max()) < float_limit:
            out[col] = series.astype("float32")
    return out


def build_timestamp(
    frame: pd.DataFrame, config: dict, dataset_name: str, *, out_col: str = "timestamp"
) -> pd.DataFrame:
    """Create the canonical datetime column from the dataset's source column."""
    cfg = config["datasets"][dataset_name]
    source = cfg["source_time_col"]
    if source not in frame.columns:
        raise KeyError(f"Source time column {source!r} not found")
    times = derive_datetime(
        frame[source],
        reference_date=config["ieee_reference_date"],
        time_unit=cfg["time_unit"],
    )
    return frame.assign(**{out_col: times})


def fill_missing_with_constants(
    frame: pd.DataFrame,
    *,
    numeric_fill: float = 0.0,
    categorical_fill: str = "Unknown",
) -> pd.DataFrame:
    """Fill NaNs with fixed constants (never with learned statistics)."""
    out = frame.copy(deep=False)
    numeric = out.select_dtypes(include=["number"]).columns
    categorical = out.select_dtypes(include=["object", "string", "category"]).columns
    out[numeric] = out[numeric].fillna(numeric_fill)
    out[categorical] = out[categorical].fillna(categorical_fill)
    return out


def clean_dataset(
    frame: pd.DataFrame,
    *,
    config: dict,
    dataset_name: str,
    time_col: str = "timestamp",
    sort: bool = True,
) -> pd.DataFrame:
    """Downcast, derive timestamp + month index, and sort chronologically."""
    out = build_timestamp(frame, config, dataset_name, out_col=time_col)
    # Kaggle CSVs often carry an unnamed row-index column; it is metadata,
    # not a transaction feature (and can look like extreme drift).
    unnamed = [c for c in out.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        out = out.drop(columns=unnamed)
    if "month" not in out.columns:
        out = add_month_index(out, time_col=time_col)
    if sort:
        out = out.sort_values(time_col, kind="stable").reset_index(drop=True)
    return downcast_numeric(out)
