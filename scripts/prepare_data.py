"""Prepare raw datasets into cleaned parquet files under ``data/interim/``.

REFERENCE runner for Role 1. Run this before any experiment script. The
provided Sparkov train/test split is ignored: both CSVs are concatenated and
the chronological split is decided later by ``src.pipeline.splits``.

    python scripts/prepare_data.py --dataset ieee_cis
    python scripts/prepare_data.py --dataset sparkov
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, get_path
from src.data.clean import clean_dataset
from src.data.load import load_raw_dataset, save_interim
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", choices=("ieee_cis", "sparkov"), required=True)
    parser.add_argument("--output-dir", type=str, default=None, help="Interim directory")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = get_config(args.config)
    set_seed(config["seed"])
    logger = setup_logger(
        "prepare_data",
        config=config,
        log_file=get_path(config, "results_dir") / "logs" / f"prepare_{args.dataset}.log",
    )

    raw = load_raw_dataset(
        config,
        args.dataset,
        include_test_file=(args.dataset == "sparkov"),
    )
    logger.info("loaded %d raw rows for %s", len(raw), args.dataset)
    clean = clean_dataset(raw, config=config, dataset_name=args.dataset)
    interim_dir = Path(args.output_dir) if args.output_dir else get_path(config, "interim_dir")
    target = interim_dir / f"{args.dataset}.parquet"
    save_interim(clean, target)

    report = clean.groupby("month").size()
    logger.info("saved %s (%d months, %d rows)", target, len(report), len(clean))
    print(f"Saved {target}")
    print(f"Rows: {len(clean)}; calendar months: {len(report)}")
    print(f"Target column: {config['datasets'][args.dataset]['target_col']}")


if __name__ == "__main__":
    main()
