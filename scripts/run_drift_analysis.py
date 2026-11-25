"""Drift analysis: PSI, KS, error-stream detectors and degradation curve.

Trains a static model on the first ``split.train_months`` and evaluates every
later calendar month. Outputs CSV tables and a degradation figure under
results/. Use ``--dataset synthetic`` for a smoke test (NOT a research result).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import get_config, get_path
from src.drift.degradation import static_degradation_curve
from src.drift.detectors import METHODS, ErrorStreamDetector
from src.drift.ks import ks_statistic
from src.drift.psi import psi_table
from src.evaluation.plots import plot_monthly_degradation
from src.models.factories import MODEL_NAMES, build_model
from src.pipeline.preprocessing import build_preprocessor, infer_feature_columns
from src.utils.inputs import load_prepared_frame
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

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


def _feature_columns(frame: pd.DataFrame, meta: dict) -> tuple[list[str], list[str]]:
    exclude = [
        meta["target_col"],
        "timestamp",
        "month",
        meta.get("key_col"),
        meta.get("source_time_col"),
    ]
    exclude += [c for c in frame.columns if any(h in c.lower() for h in _META_HINTS)]
    exclude = [c for c in dict.fromkeys(exclude) if c is not None and c in frame.columns]
    return infer_feature_columns(frame, exclude=exclude)


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
        "drift_analysis",
        config=config,
        log_file=output_root / "logs" / f"drift_analysis_{args.dataset}.log",
    )

    frame, meta = load_prepared_frame(config, dataset=args.dataset, data_path=args.data)
    numeric_cols, categorical_cols = _feature_columns(frame, meta)
    target_col = meta["target_col"]
    train_months = config["split"]["train_months"]
    months = sorted(frame["month"].unique().tolist())
    reference = frame.loc[frame["month"].isin(months[:train_months])]
    logger.info("reference months=%d, rows=%d", train_months, len(reference))

    psi_rows, ks_rows = [], []
    for month in months[train_months:]:
        current = frame.loc[frame["month"] == month]
        psi_for_month = psi_table(reference, current, numeric_cols)
        psi_for_month.insert(0, "month", month)
        psi_rows.append(psi_for_month)
        for column in numeric_cols:
            statistic, p_value = ks_statistic(reference[column], current[column])
            ks_rows.append(
                {
                    "month": month,
                    "column": column,
                    "ks_statistic": statistic,
                    "p_value": p_value,
                }
            )
    psi_table_all = pd.concat(psi_rows, ignore_index=True)
    ks_table_all = pd.DataFrame(ks_rows)

    # Error stream from one static model; detectors see only 0/1 mistakes.
    preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
        min_frequency=config["preprocessing"]["one_hot_min_frequency"],
    )
    matrix_train = preprocessor.fit_transform(reference[[*numeric_cols, *categorical_cols]])
    static_model = clone(build_model(args.model, seed=config["seed"]))
    static_model.fit(matrix_train, reference[target_col].to_numpy())

    events = []
    for method in METHODS:
        params = dict(config["retraining"].get("drift_params", {})) if method == "adwin" else {}
        detector = ErrorStreamDetector(method, **params)
        for month in months[train_months:]:
            current = frame.loc[frame["month"] == month]
            y_true = current[target_col].to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            scores = static_model.predict_proba(
                preprocessor.transform(current[[*numeric_cols, *categorical_cols]])
            )[:, 1]
            errors = ((scores >= 0.5).astype(int) != y_true).astype(int)
            for error in errors:
                if detector.update(int(error)):
                    events.append(
                        {
                            "month": month,
                            "method": method,
                            "drift_detected": True,
                        }
                    )
    events_table = pd.DataFrame(
        events, columns=["month", "method", "drift_detected"]
    )

    curve = static_degradation_curve(
        frame,
        model=build_model(args.model, seed=config["seed"]),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        train_months=train_months,
        fpr_target=config["fpr_target"],
        cost_fp=config["cost_fp"],
        cost_fn=config["cost_fn"],
        scale_numeric=(args.model == "logreg"),
        one_hot_min_frequency=config["preprocessing"]["one_hot_min_frequency"],
    )

    paths = {
        "psi": tables / f"psi_{args.dataset}.csv",
        "ks": tables / f"ks_{args.dataset}.csv",
        "events": tables / f"drift_events_{args.dataset}.csv",
        "curve": tables / f"degradation_{args.dataset}.csv",
    }
    psi_table_all.to_csv(paths["psi"], index=False)
    ks_table_all.to_csv(paths["ks"], index=False)
    events_table.to_csv(paths["events"], index=False)
    curve.to_csv(paths["curve"], index=False)
    figure = plot_monthly_degradation(
        curve,
        figures / f"degradation_{args.dataset}.png",
        title=f"Static-model degradation ({args.dataset})",
    )
    for key, path in paths.items():
        logger.info("saved %s -> %s", key, path)
    logger.info("saved figure -> %s", figure)

    if args.dataset == "synthetic":
        print("WARNING: SYNTHETIC DEMO DATA -- numbers below are NOT research results.")
    print(f"PSI table ({len(psi_table_all)} rows):")
    print(psi_table_all.groupby("column")["psi"].max().to_string())
    print(f"\nDrift events: {len(events_table)}")
    print(f"\nDegradation curve ({len(curve)} months):")
    print(curve[["month", "n_rows", "pr_auc"]].to_string(index=False))
    print(f"\nSaved outputs under {output_root}")


if __name__ == "__main__":
    main()
