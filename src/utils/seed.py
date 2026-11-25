"""Reproducibility helpers.

Setting a single seed in one place is a project requirement: every script must
call :func:`set_seed` (or :func:`set_seed_from_config`) before doing any random
work, including train/test splitting and estimator initialisation.
"""

from __future__ import annotations

import os
import random
from typing import Any, Mapping

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all random number generators used by this project.

    Parameters
    ----------
    seed:
        Non-negative integer seed. The same seed must always give the same
        splits, resampling and model initialisation.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    # Makes hash-based iteration (e.g. sets of category names) deterministic
    # for child processes started after this call.
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_seed_from_config(config: Mapping[str, Any]) -> int:
    """Read ``seed`` from a loaded config and apply it.

    Returns the seed that was applied, so callers can log it.
    """
    seed = config["seed"]
    set_seed(seed)
    return seed
