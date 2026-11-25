"""REFERENCE IMPLEMENTATION for Role 6 (Retraining) -- safe to replace."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import pandas as pd

T = TypeVar("T")


@dataclass(frozen=True)
class RetrainCost:
    """Cost of one (re)training operation."""

    strategy: str
    month: int
    train_rows: int
    seconds: float


class CostTracker:
    """Record wall-clock training time per strategy and month."""

    def __init__(self) -> None:
        self.records: list[RetrainCost] = []

    def track(
        self,
        *,
        strategy: str,
        month: int,
        train_rows: int,
        fn: Callable[[], T],
    ) -> T:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        self.records.append(RetrainCost(strategy, month, train_rows, elapsed))
        return result

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])
