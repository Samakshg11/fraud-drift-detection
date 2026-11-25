"""Shared input loading for experiment scripts.

Returns a frame with a canonical datetime ``timestamp`` and integer ``month``
column. Scripts must never assume a provided train/test split is valid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_path
from src.pipeline.splits import add_month_index, derive_datetime
from src.utils.synthetic import make_synthetic_demo


def load_prepared_frame(
    config: dict,
    *,
    dataset: str,
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load synthetic demo data or a prepared parquet/CSV frame."""
    if dataset == "synthetic":
        frame = make_synthetic_demo(config["seed"])
        return frame, {
            "target_col": "isFraud",
            "time_col": "timestamp",
            "key_col": "TransactionID",
            "source_time_col": "TransactionDT",
        }

    if dataset not in config["datasets"]:
        raise KeyError(f"Unknown dataset: {dataset}")
    dataset_cfg = config["datasets"][dataset]
    path = (
        Path(data_path)
        if data_path
        else get_path(config, "interim_dir") / f"{dataset}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Input data not found: {path}. Place raw files in data/raw/ and run "
            f"the Role 1 loading/cleaning step, or pass --data explicitly."
        )
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    source_time_col = dataset_cfg["source_time_col"]
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    else:
        frame["timestamp"] = derive_datetime(
            frame[source_time_col],
            reference_date=config["ieee_reference_date"],
            time_unit=dataset_cfg["time_unit"],
        )
    if "month" not in frame.columns:
        frame = add_month_index(frame, time_col="timestamp")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return frame, {**dataset_cfg, "time_col": "timestamp"}
