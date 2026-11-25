# Experiment Log

Purpose: make every reported number traceable to a command, a config and a dataset.

## Environment

- Python 3.11.4 (venv at `venv/`, git-ignored)
- Exact package versions: `requirements.lock.txt`
- 8 CPU cores / 8 GB RAM — real runs use **Logistic Regression** for tractability
- Seed: 42; split: 3 train months / 1 test month, expanding, `gap_days=0`
- macOS: XGBoost requires `brew install libomp` (not used in the runs below)

## Data inventory

| Dataset | Raw path | Rows | Months | Status |
|---|---|---|---|---|
| IEEE-CIS | `data/raw/ieee_cis/` | — | — | **Not available**: Kaggle account verification failed |
| Sparkov | `data/raw/sparkov/` | 1,852,394 | 24 (2019-01 → 2020-12) | Used for all experiments below |

`data/interim/sparkov.parquet` is produced by `scripts/prepare_data.py`.
The Kaggle index column `Unnamed: 0` is dropped at cleaning time.
Fraud base rate: **0.521%** (9,651 frauds).

## Run log (real data)

| Date | Command | Dataset | Outputs |
|---|---|---|---|
| 2026-11-25 | `python scripts/prepare_data.py --dataset sparkov` | Sparkov | `data/interim/sparkov.parquet` (1,852,394 rows, 24 months) |
| 2026-11-25 | `python scripts/run_leakage_comparison.py --dataset sparkov --model logreg` | Sparkov | `results/tables/leakage_comparison_sparkov.csv`, `results/figures/leakage_comparison_sparkov.png` |
| 2026-11-25 | `python scripts/run_drift_analysis.py --dataset sparkov --model logreg` | Sparkov | `psi_*.csv`, `ks_*.csv`, `drift_events_*.csv`, `degradation_*.csv/.png` |
| 2026-11-25 | `python scripts/run_retraining_experiments.py --dataset sparkov --model logreg` | Sparkov | `retraining_metrics_*.csv`, `retraining_costs_*.csv`, `retraining_pr_auc_*.png` |

All four scripts were re-run after dropping the index artifact, so the tables are consistent.

## Results

### 1. Leakage inflation (Objective 1)

| Setup | PR-AUC | Recall@1%FPR | F1 | Cost | Base rate | Rows |
|---|---|---|---|---|---|---|
| Leaky (random split, all-data fits) | **0.2275** | **0.5611** | 0.3235 | 12,152 | 0.5209% | 370,479 |
| Leakage-free (walk-forward) | **0.0061** | **0.0328** | 0.0214 | 95,276 | 0.4844% | 1,679,064 |

Leaky PR-AUC is ~37× higher. This is the project's central quantitative result.

### 2. Drift / degradation (Objective 2)

- Static model PR-AUC falls from **0.0105** (first test month) to **0.0045** (last),
  with a peak of 0.0145 — clear month-by-month degradation.
- Numeric PSI is small (max 0.00056 for `lat`, 0.00052 for `amt`), so the drift
  is mostly in label/feature relationships (concept drift), not marginal features.
- KS max statistic 0.0076; smallest p-value 0.0029 (large-n significance,
  small effect size).
- Detector events: DDM 922, ADWIN 307, Page-Hinkley 17. The high DDM count is
  likely dominated by false alarms on a 0.5%-fraud error stream at a 0.5
  decision threshold — treat as a tuning issue, not a scientific conclusion.

### 3. Retraining strategies (Objective 3)

| Strategy | PR-AUC | Recall@1%FPR | Cost | Total training time |
|---|---|---|---|---|
| Sliding window | **0.0116** | 0.0149 | **4,305** | 63 s |
| Static | 0.0102 | 0.0103 | 4,361 | 2 s |
| Weighted | 0.0083 | **0.0368** | 4,410 | 323 s |
| Periodic | 0.0064 | 0.0308 | 4,368 | 351 s |
| Drift-triggered | 0.0064 | 0.0308 | 4,368 | 348 s |

Sliding window gives the best PR-AUC at moderate cost; static is by far the
cheapest. Drift-triggered retrained on every fold because the detector kept
firing (see caveat above), so it is not a meaningful win here.

## Decisions log

| Decision | Current value | Status |
|---|---|---|
| IEEE reference date | 2017-12-01 | assumed (not used: no IEEE data) |
| Split lengths | 3 train / 1 test month | from config |
| Scheme | expanding | from config |
| Embargo (`gap_days`) | 0 | assumed — needs chargeback-lag input |
| FPR target / cost ratio | 0.01 / 10:1 | assumed |
| One-hot min frequency | 100 | added for real data (memory cap) |
| Model for reported runs | Logistic Regression | forced by 8 GB RAM |

## Open questions / TODOs

1. IEEE-CIS could not be downloaded (Kaggle verification). Need a verified teammate
   or a licensed mirror; all results would then be repeated on a second dataset.
2. Re-run results with XGBoost/Random Forest on a sampled subset if time allows.
3. Tune the drift-trigger threshold on precision/recall of alerts rather than
   using the default 0.5 decision rule.
4. Decide `gap_days` from the real chargeback lag; retraining feedback currently
   assumes labels are available immediately.
5. Fill `docs/literature_review.md` — still an empty template.

## Leakage checklist for the reported results

- [x] Chronological month splits with configured embargo
- [x] No random split outside `run_leaky_baseline()`
- [x] All learned state fitted on train fold only
- [x] Training encodings are expanding/past-only; test uses frozen mapping
- [x] Past-only aggregates use `closed="left"` windows
- [x] Metrics include PR-AUC, recall@FPR, F1 and base rate — no accuracy claim
- [x] Every table traces to a run log entry above
