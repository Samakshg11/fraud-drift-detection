"""Compare drift-aware retraining strategies month by month.

Each strategy sees the same walk-forward folds: train = all months strictly
before the test month. Use ``--dataset synthetic`` for a smoke test (NOT a
research result).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import get_config, get_path
from src.evaluation.metrics import compute_metrics_from_config
from src.evaluation.plots import plot_monthly_degradation
from src.models.factories import MODEL_NAMES, build_model
from src.pipeline.preprocessing import infer_feature_columns
from src.retraining.cost_tracker import CostTracker
from src.retraining.strategies import (
    DriftTriggeredStrategy,
    PeriodicStrategy,
    SlidingWindowStrategy,
    StaticStrategy,
    WeightedStrategy,
)
from src.utils.inputs import load_prepared_frame
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

STRATEGIES = ("static", "periodic", "sliding_window", "drift_triggered", "weighted")
_META_HINTS = (
    "timestamp",
    "unix_time",
    "transactiondt",
    "trans_date_trans_time",
    "unnamed:",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--dataset", choices=("synthetic", "ieee_cis", "sparkov"), default="synthetic"
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", choices=MODEL_NAMES, default="logreg")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def make_strategy(name: str, config: dict, *, model: object, numeric: list[str], categorical: list[str], target: str, model_name: str):
    common = dict(
        model=model,
        numeric_cols=numeric,
        categorical_cols=categorical,
        target_col=target,
        scale_numeric=(model_name == "logreg"),
        one_hot_min_frequency=config["preprocessing"]["one_hot_min_frequency"],
    )
    cfg = config["retraining"]
    if name == "static":
        return StaticStrategy(**common)
    if name == "periodic":
        return PeriodicStrategy(period_months=cfg["periodic_months"], **common)
    if name == "sliding_window":
        return SlidingWindowStrategy(window_months=cfg["sliding_window_months"], **common)
    if name == "drift_triggered":
        return DriftTriggeredStrategy(
            method=cfg["drift_method"],
            detector_params=dict(cfg.get("drift_params", {})),
            **common,
        )
    if name == "weighted":
        return WeightedStrategy(half_life_months=cfg["half_life_months"], **common)
    raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    args = build_arg_parser().parse_args()
    config = get_config(args.config)
    set_seed(config["seed"])
    output_root = Path(args.output_dir) if args.output_dir else get_path(config, "results_dir")
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "retraining_experiments",
        config=config,
        log_file=output_root / "logs" / f"retraining_{args.dataset}.log",
    )

    frame, meta = load_prepared_frame(config, dataset=args.dataset, data_path=args.data)
    exclude = [
        meta["target_col"],
        "timestamp",
        "month",
        meta.get("key_col"),
        meta.get("source_time_col"),
    ]
    exclude += [c for c in frame.columns if any(h in c.lower() for h in _META_HINTS)]
    exclude = [c for c in dict.fromkeys(exclude) if c is not None and c in frame.columns]
    numeric, categorical = infer_feature_columns(frame, exclude=exclude)
    target = meta["target_col"]
    months = sorted(frame["month"].unique().tolist())
    train_months = config["split"]["train_months"]

    rows: list[dict] = []
    tracker = CostTracker()
    for name in STRATEGIES:
        strategy = make_strategy(
            name,
            config,
            model=build_model(args.model, seed=config["seed"]),
            numeric=numeric,
            categorical=categorical,
            target=target,
            model_name=args.model,
        )
        for month in months[train_months:]:
            train_df = frame.loc[frame["month"] < month]
            test_df = frame.loc[frame["month"] == month]
            y_true = test_df[target].to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            tracker.track(
                strategy=name,
                month=month,
                train_rows=len(train_df),
                fn=lambda: strategy.fit(train_df),
            )
            scores = strategy.predict_proba(test_df)
            row = compute_metrics_from_config(y_true, scores, config, setup=name)
            row["month"] = month
            rows.append(row)
            # Feedback hook: only drift_triggered uses it.
            strategy.update(y_true, scores)
        logger.info(
            "%s: %d refits, last training rows=%d",
            name,
            strategy.refit_count_,
            strategy.last_train_rows_,
        )

    metrics_table = pd.DataFrame(rows)
    costs_table = tracker.to_frame()
    metrics_path = tables / f"retraining_metrics_{args.dataset}.csv"
    costs_path = tables / f"retraining_costs_{args.dataset}.csv"
    metrics_table.to_csv(metrics_path, index=False)
    costs_table.to_csv(costs_path, index=False)
    figure = plot_monthly_degradation(
        metrics_table,
        figures / f"retraining_pr_auc_{args.dataset}.png",
        series_col="strategy",
        metric="pr_auc",
        title=f"Retraining strategies ({args.dataset})",
    )

    if args.dataset == "synthetic":
        print("WARNING: SYNTHETIC DEMO DATA -- numbers below are NOT research results.")
    summary = (
        metrics_table.groupby("setup")[["pr_auc", "recall_at_fpr", "f1", "cost_at_fpr"]]
        .mean()
        .round(4)
    )
    print(summary.to_string())
    print(f"\nSaved: {metrics_path}\nSaved: {costs_path}\nSaved: {figure}")


if __name__ == "__main__":
    main()
