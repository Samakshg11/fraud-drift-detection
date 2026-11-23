"""Quantify metric inflation from the classic leakage mistakes.

Runs two baselines on the same frame:

* ``leaky``      - random split, encoders + preprocessing fitted on all rows,
                   optional full-timeline group aggregates (INVALID);
* ``leakage_free`` - walk-forward month folds, past-only training encodings,
                   frozen train-period mappings, train-only preprocessing.

Metrics and plotting come from the Role 4 reference modules
``src.evaluation.metrics`` / ``src.evaluation.plots``.

Examples
--------
Synthetic smoke test (numbers are NOT research results)::

    python scripts/run_leakage_comparison.py --dataset synthetic

Real data (after Role 1 has produced an interim parquet)::

    python scripts/run_leakage_comparison.py --dataset ieee_cis \
        --data data/interim/ieee_cis.parquet --model xgboost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/<name>.py` from the repo root without PYTHONPATH setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import get_config, get_path
from src.evaluation.metrics import compute_metrics_from_config, metrics_table
from src.evaluation.plots import plot_metric_comparison
from src.models.factories import MODEL_NAMES, build_model
from src.pipeline.leaky_baseline import run_leakage_free_baseline, run_leaky_baseline
from src.pipeline.preprocessing import infer_feature_columns
from src.utils.inputs import load_prepared_frame
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

# Metadata columns that must never be used as plain model features.
_META_HINTS = ("timestamp", "unix_time", "transactiondt", "trans_date_trans_time")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "ieee_cis", "sparkov"),
        default="synthetic",
    )
    parser.add_argument("--data", type=str, default=None, help="CSV or parquet input")
    parser.add_argument("--model", choices=MODEL_NAMES, default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2, help="Leaky random split size")
    parser.add_argument(
        "--include-global-aggregates",
        action="store_true",
        help="Add INVALID full-timeline group aggregates to the leaky baseline only",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def save_outputs(table: pd.DataFrame, dataset: str, output_dir: Path) -> tuple[Path, Path]:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    csv_path = tables / f"leakage_comparison_{dataset}.csv"
    fig_path = figures / f"leakage_comparison_{dataset}.png"
    table.to_csv(csv_path, index=False)
    plot_metric_comparison(
        table,
        ["pr_auc", "recall_at_fpr"],
        fig_path,
        title=f"Leaky vs leakage-free ({dataset})",
    )
    return csv_path, fig_path


def main() -> None:
    args = build_arg_parser().parse_args()
    config = get_config(args.config)
    set_seed(config["seed"])

    output_root = Path(args.output_dir) if args.output_dir else get_path(config, "results_dir")
    logs_dir = output_root / "logs"
    logger = setup_logger(
        "leakage_comparison",
        config=config,
        log_file=logs_dir / f"leakage_comparison_{args.dataset}.log",
    )

    frame, meta = load_prepared_frame(
        config, dataset=args.dataset, data_path=args.data
    )
    target_col = meta["target_col"]
    time_col = meta["time_col"]
    key_col = meta.get("key_col")

    exclude = [target_col, time_col, "month", key_col, meta.get("source_time_col")]
    exclude += [c for c in frame.columns if any(h in c.lower() for h in _META_HINTS)]
    exclude = [c for c in dict.fromkeys(exclude) if c is not None and c in frame.columns]
    numeric_cols, categorical_cols = infer_feature_columns(frame, exclude=exclude)

    cardinalities = {c: int(frame[c].nunique(dropna=True)) for c in categorical_cols}
    eligible = [c for c, k in cardinalities.items() if 1 < k <= 50]
    target_encoded_cols = sorted(eligible, key=lambda c: cardinalities[c])[:1]
    logger.info(
        "rows=%d, numeric=%d, categorical=%d, target-encoded=%s",
        len(frame), len(numeric_cols), len(categorical_cols), target_encoded_cols,
    )

    common = dict(
        target_col=target_col,
        time_col=time_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_encoded_cols=target_encoded_cols,
        model=build_model(args.model, seed=config["seed"]),
        scale_numeric=(args.model == "logreg"),
    )
    group_col = categorical_cols[0] if categorical_cols and numeric_cols else None
    value_col = numeric_cols[0] if numeric_cols else None
    if not (args.include_global_aggregates and group_col and value_col):
        group_col = value_col = None

    leaky = run_leaky_baseline(
        frame,
        **common,
        seed=config["seed"],
        test_size=args.test_size,
        group_col=group_col,
        value_col=value_col,
    )
    clean = run_leakage_free_baseline(
        frame,
        **common,
        month_col="month",
        train_months=config["split"]["train_months"],
        test_months=config["split"]["test_months"],
        gap_days=config["split"]["gap_days"],
        scheme=config["split"]["scheme"],
        sliding_window_months=config["split"]["sliding_window_months"],
    )
    for summary in clean.fold_summaries:
        logger.info("fold %s", summary)

    table = metrics_table(
        [
            compute_metrics_from_config(r.y_true, r.y_score, config, setup=r.label)
            for r in (leaky, clean)
        ]
    )
    csv_path, fig_path = save_outputs(table, args.dataset, output_root)
    logger.info("Saved %s and %s", csv_path, fig_path)

    if args.dataset == "synthetic":
        print("WARNING: SYNTHETIC DEMO DATA -- numbers below are NOT research results.")
    print(table.to_string(index=False))
    print(f"\nSaved: {csv_path}\nSaved: {fig_path}")


if __name__ == "__main__":
    main()
