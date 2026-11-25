# Leakage-Free and Drift-Aware Credit Card Fraud Detection

Machine-learning research project on long-horizon transaction data (IEEE-CIS, ~6 months;
Sparkov, Jan 2019–Dec 2020). The project:

1. builds a **leakage-free** pipeline with chronological month splits and walk-forward validation;
2. quantifies how much a **deliberately leaky** setup inflates metrics;
3. measures month-by-month degradation of a static model;
4. compares drift-aware retraining strategies (static, periodic, sliding window,
   drift-triggered ADWIN/DDM/Page-Hinkley, recency-weighted).

Classic ML only (Logistic Regression, Random Forest, XGBoost). **Accuracy is never used.**

## Setup

Use Python 3.11 (recommended; 3.14 lacks wheels for some dependencies).

```bash
git clone https://github.com/Samakshg11/fraud-drift-detection.git
cd fraud-drift-detection
git checkout dev
git checkout -b feature/<your-name>
python3.11 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt     # reproducible: pip install -r requirements.lock.txt
```

On macOS, XGBoost needs OpenMP: `brew install libomp`. XGBoost imports lazily, so
all other scripts work without it.

## Data (never committed)

Download into `data/raw/`:

- IEEE-CIS: `data/raw/ieee_cis/train_transaction.csv`
- Sparkov: `data/raw/sparkov/fraudTrain.csv` and `fraudTest.csv`

Datasets are git-ignored. The provided Sparkov train/test split is **not** used —
both files are concatenated and sorted by time in `scripts/prepare_data.py`.

## Run order

Run every script from the repository root with the venv activated.

```bash
# 0. Smoke tests (synthetic data, NOT research results)
venv/bin/python -m pytest -q
venv/bin/python scripts/run_leakage_comparison.py --dataset synthetic

# 1. Data preparation (Role 1) -> data/interim/<dataset>.parquet
venv/bin/python scripts/prepare_data.py --dataset ieee_cis
venv/bin/python scripts/prepare_data.py --dataset sparkov

# 2. Leaky vs leakage-free comparison (Role 2)
venv/bin/python scripts/run_leakage_comparison.py --dataset ieee_cis --model xgboost
#   optionally add: --include-global-aggregates   (deliberately invalid feature leak)

# 3. Drift analysis: PSI, KS, error-stream detectors, degradation curve (Role 5)
venv/bin/python scripts/run_drift_analysis.py --dataset ieee_cis --model xgboost

# 4. Retraining strategies (Role 6)
venv/bin/python scripts/run_retraining_experiments.py --dataset ieee_cis --model xgboost
```

Outputs are written to `results/tables/*.csv`, `results/figures/*.png` and
`results/logs/*.log`. Synthetic artifacts are git-ignored so they can never be
mistaken for experimental results. Metrics are computed from real runs only;
this README reports none.

## Leakage rules (the contract of this project)

1. **Chronological month splits only** (`src/pipeline/splits.py`), with an optional
   `gap_days` embargo for late fraud labels. `max(train_time) < min(test_time)` is asserted.
2. **Everything learned is fitted on the training fold only** — imputation, scaling,
   one-hot vocabularies, encoders, resampling, feature selection, thresholds.
3. **Time-aware encoders** (`src/pipeline/encoders.py`) use strictly earlier rows
   for training values and a frozen train-period mapping for test rows.
4. **Past-only aggregates** (`src/features/aggregates.py`) use `rolling(closed="left")`
   with the current row excluded.
5. **Walk-forward, never shuffled KFold**, for tuning and evaluation.
6. **Resampling only on training folds.**
7. **Primary metrics**: PR-AUC, recall at fixed FPR, F1/precision/recall, configurable
   cost metric, and always the fraud base rate.

`src/pipeline/leaky_baseline.py` is the only place where these rules are intentionally
broken, and only to measure the resulting inflation.

## Configuration

All settings live in `config.yaml` (seed, IEEE reference date, split lengths and
scheme, embargo, FPR target, costs, paths, dataset metadata, retraining windows).
`src/config.py` validates them at load time; an invalid split scheme fails fast.

## Role ownership

| Role | Area | Modules |
|---|---|---|
| 1. Data Engineer | `src/data/` | `load.py`, `clean.py` (reference impls + `scripts/prepare_data.py`) |
| 2. Leakage-Free Pipeline Lead | `src/pipeline/` | splits, preprocessing, encoders, leaky baseline, tests |
| 3. Feature Engineering | `src/features/` | past-only aggregates, time features (reference impls) |
| 4. Modeling | `src/models/`, `src/evaluation/` | factories, metrics, plots (reference impls) |
| 5. Drift Analysis | `src/drift/` | PSI, KS, ADWIN/DDM/Page-Hinkley, degradation (reference impls) |
| 6. Retraining + Report | `src/retraining/`, `paper/` | strategies, cost tracking, scripts, final paper |

Modules in Roles 1, 3, 4, 5 and 6 are clearly marked **REFERENCE IMPLEMENTATION —
safe to replace**. The owning teammate may replace them without touching Role 2.

## Folder structure

- `src/` — all code; `notebooks/` should only call functions from `src/`
- `tests/` — synthetic-data tests, including explicit leakage proofs
- `results/` — figures, tables and logs (generated)
- `paper/`, `docs/` — report sections, experiment log and literature review

## Git rules

- `main` is protected; never push directly to `main` or `dev`.
- Work on `feature/<name>` from `dev` and open a PR with a teammate review.
- Never commit raw/interim/processed data, models, logs or secrets.

See `docs/experiment_log.md` for the run log and outstanding decisions.
