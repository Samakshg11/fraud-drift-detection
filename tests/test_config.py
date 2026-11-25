"""Tests for configuration loading, legacy migration and validation."""

from __future__ import annotations

import copy

import pytest
import yaml

from src.config import (
    find_config_path,
    get_config,
    get_path,
    load_config,
    repo_root,
    reset_config_cache,
)
from src.utils.logger import setup_logger
from src.utils.seed import set_seed, set_seed_from_config

import numpy as np


def _write_config(tmp_path, raw: dict) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def _valid_raw() -> dict:
    # Start from the real config so tests stay in sync with the repository.
    return copy.deepcopy(yaml.safe_load(find_config_path().read_text(encoding="utf-8")))


def test_repository_config_is_valid():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["split"]["scheme"] in {"expanding", "sliding"}
    assert 0 < cfg["fpr_target"] < 1
    assert cfg["datasets"]["ieee_cis"]["target_col"] == "isFraud"
    assert cfg["datasets"]["sparkov"]["target_col"] == "is_fraud"


def test_legacy_top_level_sliding_window_is_migrated(tmp_path):
    raw = _valid_raw()
    raw["sliding_window_months"] = raw["split"].pop("sliding_window_months")
    cfg = load_config(_write_config(tmp_path, raw))
    assert cfg["split"]["sliding_window_months"] == 3
    assert "sliding_window_months" not in cfg


def test_invalid_scheme_and_fpr_are_rejected(tmp_path):
    raw = _valid_raw()
    raw["split"]["scheme"] = "random"  # forbidden: no random splits
    with pytest.raises(ValueError, match="split.scheme"):
        load_config(_write_config(tmp_path, raw))

    raw = _valid_raw()
    raw["fpr_target"] = 1.5
    with pytest.raises(ValueError, match="fpr_target"):
        load_config(_write_config(tmp_path, raw))


def test_missing_key_is_rejected(tmp_path):
    raw = _valid_raw()
    del raw["split"]["gap_days"]
    with pytest.raises(ValueError, match="gap_days"):
        load_config(_write_config(tmp_path, raw))


def test_get_path_is_absolute_and_under_repo():
    cfg = load_config()
    path = get_path(cfg, "tables_dir")
    assert path.is_absolute()
    assert path.is_relative_to(repo_root())
    assert path.name == "tables"


def test_get_config_cache_and_reset(tmp_path):
    reset_config_cache()
    path = _write_config(tmp_path, _valid_raw())
    first = get_config(path)
    assert get_config(path) == first
    # get_config returns a defensive deep copy: mutating it cannot poison cache.
    first["seed"] = -999
    assert get_config(path)["seed"] == 42
    changed = get_config(path, reload=True)
    assert changed["seed"] == 42
    reset_config_cache()
    assert get_config(path)["seed"] == 42


def test_set_seed_is_reproducible():
    set_seed(7)
    first = np.random.random(5)
    set_seed(7)
    second = np.random.random(5)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        set_seed(-1)


def test_set_seed_from_config_uses_yaml_seed():
    assert set_seed_from_config(load_config()) == 42


def test_logger_reconfiguration_does_not_duplicate_handlers(tmp_path):
    logger = setup_logger("test-pipeline", config=load_config())
    n_handlers = len(logger.handlers)
    logger = setup_logger("test-pipeline", config=load_config())
    assert len(logger.handlers) == n_handlers

    log_path = tmp_path / "logs" / "run.log"
    logger = setup_logger("test-pipeline", config=load_config(), log_file=log_path)
    logger.info("hello")
    assert log_path.is_file()
    assert "hello" in log_path.read_text(encoding="utf-8")
