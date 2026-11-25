"""REFERENCE IMPLEMENTATION for Role 4 (Modeling) -- safe to replace."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_comparison(
    table: pd.DataFrame,
    metrics: Sequence[str],
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Grouped bar chart comparing setups across metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        ax.bar(table["setup"], table[metric], color=["#c44e52", "#4c72b0"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_monthly_degradation(
    table: pd.DataFrame,
    path: str | Path,
    *,
    month_col: str = "month",
    metric: str = "pr_auc",
    series_col: str | None = None,
    title: str | None = None,
) -> Path:
    """Line chart of a metric against calendar month (for drift analysis)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if series_col and series_col in table.columns:
        for label, group in table.groupby(series_col):
            ax.plot(group[month_col], group[metric], marker="o", label=str(label))
        ax.legend()
    else:
        ax.plot(table[month_col], table[metric], marker="o")
    ax.set_xlabel("Month index")
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
