"""Quantify metric inflation from the classic leakage mistakes.

Runs two baselines on the same frame:

* ``leaky``      - random split, encoders + preprocessing fitted on all rows,
                   optional full-timeline group aggregates (INVALID);
* ``leakage_free`` - walk-forward month folds, past-only training encodings,
                   frozen train-period mappings, train-only preprocessing.

The metric helpers here are intentionally local and small. Role 4 will replace
them with ``src/evaluation/metrics.py`` in Phase 3.

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from src.config import get_config, get_path
from src.pipeline.leaky_baseline import BaselineResult, run_leakage_free_baseline, run_leaky_baseline
from src.pipeline.preprocessing import infer_feature_columns
from src.pipeline.splits import add_month_index, derive_datetime
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
    parser.add_argument("--model", choices=("logreg", "rf", "xgboost"), default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2, help="Leaky random split size")
    parser.add_argument(
        "--include-global-aggregates",
        action="store_true",
        help="Add INVALID full-timeline group aggregates to the leaky baseline only",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def make_synthetic_demo(seed: int) -> pd.DataFrame:
    """Generate a clearly-labelled synthetic frame; never a research result."""
    rng = np.random.default_rng(seed)
    n_per_month, months = 120, 12
    month_offsets = np.repeat(np.arange(months), n_per_month)
    base = pd.Timestamp("2019-01-01").to_period("M")
    timestamps = []
    for position, offset in enumerate(month_offsets):
        within = position % n_per_month
        timestamps.append(
            (base + int(offset)).to_timestamp()
            + pd.Timedelta(days=within % 28, hours=(within * 7) % 24)
        )
    times = pd.Series(pd.to_datetime(timestamps))
    n = len(times)
    amount = np.round(rng.lognormal(3.0, 0.6, n), 2)
    merchant = rng.choice(["m_a", "m_b", "m_c", "m_d"], n)
    rate = 0.005 + 0.02 * (month_offsets / (months - 1))
    logit = np.log(rate / (1 - rate)) + 0.9 * (merchant == "m_a")
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    frame = pd.DataFrame(
        {
            "timestamp": times,
            "TransactionID": np.arange(n),
            "TransactionAmt": amount,
            "card1": rng.integers(1000, 1100, n).astype(float),
            "ProductCD": merchant,
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "isFraud": y,
        }
    )
    return add_month_index(frame, time_col="timestamp").sort_values(
        "timestamp", kind="stable"
    )


def load_frame(args: argparse.Namespace, config: dict) -> tuple[pd.DataFrame, dict]:
    """Load the requested frame and return it with datetime + month columns."""
    if args.dataset == "synthetic":
        return make_synthetic_demo(config["seed"]), {
            "target_col": "isFraud",
            "time_col": "timestamp",
            "key_col": "TransactionID",
            "source_time_col": "TransactionDT",
        }

    dataset_cfg = config["datasets"][args.dataset]
    path = Path(args.data) if args.data else get_path(config, "interim_dir") / f"{args.dataset}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Input data not found: {path}. Place raw datasets in data/raw/ and "
            f"run the Role 1 loading script, or pass --data explicitly."
        )
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    target_col = dataset_cfg["target_col"]
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


def build_model(name: str, seed: int):
    """Unfitted estimator; baselines clone it per fold."""
    if name == "logreg":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    # Imported lazily so logreg/RF smoke tests work even if libomp is missing.
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, fpr_target: float
) -> tuple[float, float, float]:
    """Return (recall, achieved_fpr, threshold) at the requested FPR ceiling."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    # Keep every threshold: drop_intermediate=True can remove all low-FPR points
    # on small/imbalanced test sets, which would make recall@FPR undefined.
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= fpr_target)
    if len(eligible) == 0:
        # No score achieves the FPR ceiling: predict no positives at all.
        return 0.0, 0.0, float(np.max(y_score) + 1.0)
    best = int(eligible[np.argmax(tpr[eligible])])
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):  # roc_curve's first point sits at +inf
        threshold = float(np.max(y_score) + 1.0)
    return float(tpr[best]), float(fpr[best]), threshold


def compute_metrics(result: BaselineResult, config: dict) -> dict:
    """Compute the project's primary metrics for one baseline result."""
    y_true = result.y_true
    y_score = result.y_score
    recall, achieved_fpr, threshold = recall_at_fpr(
        y_true, y_score, config["fpr_target"]
    )
    if np.isfinite(threshold):
        y_pred = (y_score >= threshold).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        recall_at_thr = float(recall_score(y_true, y_pred, zero_division=0))
        false_pos = int(((y_pred == 1) & (y_true == 0)).sum())
        false_neg = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = false_pos * config["cost_fp"] + false_neg * config["cost_fn"]
    else:
        precision = f1 = recall_at_thr = cost = float("nan")

    return {
        "setup": result.label,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_fpr": recall,
        "achieved_fpr": achieved_fpr,
        "precision": precision,
        "recall": recall_at_thr,
        "f1": f1,
        "cost_at_fpr": cost,
        "fraud_base_rate": float(y_true.mean()),
        "n_rows": result.n_rows,
    }


def save_outputs(table: pd.DataFrame, dataset: str, output_dir: Path) -> tuple[Path, Path]:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    csv_path = tables / f"leakage_comparison_{dataset}.csv"
    fig_path = figures / f"leakage_comparison_{dataset}.png"
    table.to_csv(csv_path, index=False)

    metrics_to_plot = ["pr_auc", "recall_at_fpr"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(10, 4), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics_to_plot):
        ax.bar(table["setup"], table[metric], color=["#c44e52", "#4c72b0"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Leaky vs leakage-free ({dataset})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
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

    frame, meta = load_frame(args, config)
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
        model=build_model(args.model, config["seed"]),
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

    table = pd.DataFrame([compute_metrics(r, config) for r in (leaky, clean)])
    csv_path, fig_path = save_outputs(table, args.dataset, output_root)
    logger.info("Saved %s and %s", csv_path, fig_path)

    if args.dataset == "synthetic":
        print("WARNING: SYNTHETIC DEMO DATA -- numbers below are NOT research results.")
    print(table.to_string(index=False))
    print(f"\nSaved: {csv_path}\nSaved: {fig_path}")


if __name__ == "__main__":
    main()
