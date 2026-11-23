# Experiment Log

Purpose: make every reported number traceable to a command, a config and a dataset.
**No experimental results have been recorded yet.** The datasets are not in the
repository; all scripts have only been smoke-tested on clearly labelled synthetic
data. Fill this file in after running on real data.

## Environment

- Python 3.11.4 (venv at `venv/`, git-ignored)
- Exact package versions: `requirements.lock.txt` (generated from the working venv)
- macOS: XGBoost requires `brew install libomp`
- Seed: see `config.yaml` (`seed: 42`)

## Data inventory

| Dataset | Raw path | Rows | Months | Status |
|---|---|---|---|---|
| IEEE-CIS | `data/raw/ieee_cis/` | unknown | unknown | not placed |
| Sparkov | `data/raw/sparkov/` | unknown | unknown | not placed |

`TransactionDT` is a seconds offset from an unknown reference; the assumed
reference in `config.yaml` is `2017-12-01`.

## Run log

| Date | Command | Dataset | Config changes | Outputs | Notes |
|---|---|---|---|---|---|
| — | `pytest -q` | synthetic | none | 55 tests passed | local verification only |
| — | `run_leakage_comparison.py --dataset synthetic` | synthetic | none | git-ignored | smoke test; NOT a result |
| — | `run_drift_analysis.py --dataset synthetic` | synthetic | none | git-ignored | smoke test; NOT a result |
| — | `run_retraining_experiments.py --dataset synthetic` | synthetic | none | git-ignored | smoke test; NOT a result |
| — | — | IEEE-CIS | — | — | **not yet run** |
| — | — | Sparkov | — | — | **not yet run** |

## Decisions log

| Decision | Current value | Status | Owner |
|---|---|---|---|
| IEEE reference date | 2017-12-01 | assumed | all |
| Split lengths | 3 train / 1 test months | from initial config | Role 2 |
| Scheme | expanding (sliding supported) | assumed | Role 2 |
| Embargo (`gap_days`) | 0 | assumed — needs chargeback-lag input | all |
| FPR target | 0.01 | assumed | Role 4 |
| Cost ratio (FN:FP) | 10:1 | assumed | all |
| Retraining windows | periodic 1 month, sliding 3 months, half-life 3 | assumed | Role 6 |
| Drift trigger | ADWIN on error stream | assumed | Role 5 |

## Open questions / TODOs

1. **Role 1**: confirm final interim parquet path/name and column dtypes after EDA.
2. **Role 3**: choose which past-only aggregates and windows to use; 24h/7D are placeholders.
3. **Role 4**: confirm FPR/cost settings and whether to tune hyperparameters by walk-forward CV.
4. **Role 5**: decide categorical drift measure and PSI action threshold.
5. **Role 6**: drift-trigger feedback currently uses a fixed 0.5 threshold and assumes
   labels are available immediately. With real chargeback lag, feedback should be
   delayed by `gap_days` — decide before final experiments.
6. **All**: confirm cost ratio and embargo; pin dependencies via `requirements.lock.txt`.

## Leakage checklist before any reported result

- [ ] Splits are chronological by month, with the configured embargo.
- [ ] No random split outside `run_leaky_baseline()`.
- [ ] All learned state fitted on train fold only (encoders, imputer, scaler, one-hot).
- [ ] Encoding of training rows is expanding/past-only; test uses the frozen mapping.
- [ ] Aggregates use `closed="left"` windows excluding the current row.
- [ ] Resampling (if any) applied only to training folds.
- [ ] Metrics include PR-AUC, recall@FPR, F1, base rate — never accuracy alone.
- [ ] Every table in `results/` traces to a run log entry above.
