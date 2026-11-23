"""Load, migrate and validate the YAML configuration.

Design notes
------------
* ``load_config`` always returns a fresh deep copy; no caller can mutate a
  shared singleton by accident.
* ``get_config`` caches by resolved file path and is what scripts should use.
* The original repository config had ``sliding_window_months`` at the top
  level. ``_migrate_legacy_keys`` keeps that layout working so teammates who
  still have the old config are not broken.
"""

from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_NAME = "config.yaml"
_ENV_CONFIG_PATH = "FRAUD_CONFIG"

_SPLIT_KEYS = (
    "time_col",
    "train_months",
    "test_months",
    "gap_days",
    "scheme",
    "sliding_window_months",
)
_DATASET_KEYS = ("raw_dir", "source_time_col", "time_unit", "target_col", "key_col")

_CONFIG_CACHE: dict[Path, dict[str, Any]] = {}


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def find_config_path(path: str | Path | None = None) -> Path:
    """Locate the config file.

    Resolution order: explicit argument, ``FRAUD_CONFIG`` environment variable,
    then ``config.yaml`` at the repository root.
    """
    if path is not None:
        candidate = Path(path).expanduser()
    elif _ENV_CONFIG_PATH in os.environ:
        candidate = Path(os.environ[_ENV_CONFIG_PATH]).expanduser()
    else:
        candidate = repo_root() / _DEFAULT_CONFIG_NAME

    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return candidate


def _migrate_legacy_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Move pre-existing top-level keys into their current sections."""
    cfg = copy.deepcopy(raw)
    split = cfg.setdefault("split", {})
    if "sliding_window_months" in cfg and "sliding_window_months" not in split:
        split["sliding_window_months"] = cfg.pop("sliding_window_months")
    return cfg


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {context}.{key}")
    return mapping[key]


def _as_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _as_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _validate(cfg: dict[str, Any]) -> None:
    """Raise a descriptive ValueError if the config is unusable."""
    if "seed" not in cfg:
        raise ValueError("Missing required config key: seed")
    _as_positive_int(cfg["seed"], "seed")

    reference_date = _require(cfg, "ieee_reference_date", "config")
    try:
        datetime.strptime(str(reference_date), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"ieee_reference_date must be YYYY-MM-DD, got {reference_date!r}"
        ) from exc

    split = cfg.get("split")
    if not isinstance(split, dict):
        raise ValueError("config.split must be a mapping")
    for key in _SPLIT_KEYS:
        _require(split, key, "split")
    if not isinstance(split["time_col"], str) or not split["time_col"]:
        raise ValueError("split.time_col must be a non-empty string")
    _as_positive_int(split["train_months"], "split.train_months")
    _as_positive_int(split["test_months"], "split.test_months")
    _as_positive_int(split["sliding_window_months"], "split.sliding_window_months")
    gap_days = split["gap_days"]
    if not isinstance(gap_days, int) or isinstance(gap_days, bool) or gap_days < 0:
        raise ValueError(f"split.gap_days must be a non-negative int, got {gap_days!r}")
    if split["scheme"] not in ("expanding", "sliding"):
        raise ValueError(
            f"split.scheme must be 'expanding' or 'sliding', got {split['scheme']!r}"
        )

    fpr_target = _as_number(_require(cfg, "fpr_target", "config"), "fpr_target")
    if not 0.0 < fpr_target < 1.0:
        raise ValueError(f"fpr_target must be in (0, 1), got {fpr_target!r}")
    _as_number(_require(cfg, "cost_fp", "config"), "cost_fp", positive=True)
    _as_number(_require(cfg, "cost_fn", "config"), "cost_fn", positive=True)

    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("config.datasets must be a mapping")
    for name in ("ieee_cis", "sparkov"):
        if name not in datasets:
            raise ValueError(f"Missing required config key: datasets.{name}")
        for key in _DATASET_KEYS:
            _require(datasets[name], key, f"datasets.{name}")

    if not isinstance(cfg.get("paths", {}), dict):
        raise ValueError("config.paths must be a mapping")


def load_config(path: str | Path | None = None, *, validate: bool = True) -> dict[str, Any]:
    """Load the YAML config as a fresh dictionary.

    Parameters
    ----------
    path:
        Optional explicit config path (see :func:`find_config_path`).
    validate:
        Set to ``False`` only when inspecting a deliberately invalid config
        in tests.
    """
    config_path = find_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    cfg = _migrate_legacy_keys(raw)
    if validate:
        _validate(cfg)
    cfg["_config_path"] = str(config_path)
    return cfg


def get_config(
    path: str | Path | None = None, *, reload: bool = False
) -> dict[str, Any]:
    """Return a cached config dict (use ``reload=True`` after editing the file)."""
    config_path = find_config_path(path)
    if reload or config_path not in _CONFIG_CACHE:
        _CONFIG_CACHE[config_path] = load_config(config_path)
    return copy.deepcopy(_CONFIG_CACHE[config_path])


def reset_config_cache() -> None:
    """Clear the cache. Mainly useful in tests."""
    _CONFIG_CACHE.clear()


def get_path(config: dict[str, Any], key: str) -> Path:
    """Resolve ``config.paths[key]`` to an absolute path under the repo root."""
    relative = _require(config.get("paths", {}), key, "paths")
    path = Path(str(relative)).expanduser()
    return path if path.is_absolute() else repo_root() / path
