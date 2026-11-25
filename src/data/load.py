"""REFERENCE IMPLEMENTATION for Role 1 (Data Engineer) -- safe to replace.

Raw I/O only. Transformations (timestamps, downcasting, cleaning) live in
``src/data/clean.py``. Raw files are never committed to the repository.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import repo_root


def dataset_raw_dir(config: dict, dataset_name: str) -> Path:
    """Absolute path to a dataset's raw directory from config."""
    relative = Path(config["datasets"][dataset_name]["raw_dir"])
    return relative if relative.is_absolute() else repo_root() / relative


def read_csv_memory_aware(
    path: str | Path,
    *,
    usecols: list[str] | None = None,
    dtype: dict[str, str] | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Read a raw CSV, optionally in chunks.

    ``low_memory=False`` avoids mixed-dtype inference warnings on IEEE-CIS.
    Use ``usecols`` to skip columns early when memory is tight.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
        chunksize=chunksize,
    )
    if chunksize is None:
        return reader
    return pd.concat(reader, ignore_index=True)


def load_raw_dataset(
    config: dict,
    dataset_name: str,
    *,
    include_test_file: bool = False,
) -> pd.DataFrame:
    """Load the raw train file (and optionally the provided test file).

    For Sparkov we MUST concatenate ``fraudTrain.csv`` and ``fraudTest.csv``
    and ignore the provided split: chronological evaluation should decide the
    split, not the dataset distribution.
    """
    if dataset_name not in config["datasets"]:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    cfg = config["datasets"][dataset_name]
    raw_dir = dataset_raw_dir(config, dataset_name)
    frame = read_csv_memory_aware(raw_dir / cfg["train_file"])
    if include_test_file and "test_file" in cfg:
        extra = read_csv_memory_aware(raw_dir / cfg["test_file"])
        frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def load_ieee_cis(config: dict) -> pd.DataFrame:
    """Load IEEE-CIS transaction table (``TransactionDT`` is a seconds offset)."""
    return load_raw_dataset(config, "ieee_cis")


def load_sparkov(config: dict) -> pd.DataFrame:
    """Load Sparkov, concatenating the provided train/test CSVs."""
    return load_raw_dataset(config, "sparkov", include_test_file=True)


def save_interim(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a cleaned frame to parquet under ``data/interim/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_interim(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
