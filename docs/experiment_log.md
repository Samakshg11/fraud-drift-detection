# Experiment Log

Purpose: make every reported number traceable to a command, a config and a dataset.

## Environment

- Python 3.11.4 (venv at `venv/`, git-ignored)
- Exact package versions: `requirements.lock.txt`
- 8 CPU cores / 8 GB RAM — all reported runs use **Logistic Regression** for tractability
- Seed 42; split: 3 train months / 1 test month, expanding, `gap_days=0`
- macOS: XGBoost needs `brew install libomp` (not used in reported runs)

## Data inventory

| Dataset | Raw path | Rows | Months | Fraud base rate | Status |
|---|---|---|---|---|---|
| IEEE-CIS | `data/raw/ieee_cis/train_transaction.csv` | 590,540 | 7 (Dec 2017 – Jun 2018) | 3.50% | Used |
| Sparkov | `data/raw/sparkov/` | 1,852,394 | 24 (Jan 2019 – Dec 2020) | 0.521% | Used |

IEEE `TransactionDT` is a seconds offset; the assumed reference `2017-12-01` in
`config.yaml` yields 7 months. The final month (Jun 2018) has only 2,754 rows,
so its monthly metric is noisy. Kaggle CSV index columns (`Unnamed: 0`) are dropped.

## Run log

| Date | Command | Dataset | Outputs |
|---|---|---|---|
| 2026-11-25 | `prepare_data.py --dataset sparkov` | Sparkov | `data/interim/sparkov.parquet` |
| 2026-11-25 | `run_leakage_comparison.py --dataset sparkov --model logreg` | Sparkov | `leakage_comparison_sparkov.csv/.png` |
| 2026-11-25 | `run_drift_analysis.py --dataset sparkov --model logreg` | Sparkov | `psi/ks/drift_events/degradation_sparkov.csv`, `degradation_sparkov.png` |
| 2026-11-25 | `run_retraining_experiments.py --dataset sparkov --model logreg` | Sparkov | `retraining_metrics/costs_sparkov.csv`, `retraining_pr_auc_sparkov.png` |
| 2026-11-25 | `prepare_data.py --dataset ieee_cis` | IEEE-CIS | `data/interim/ieee_cis.parquet` |
| 2026-11-25 | `run_leakage_comparison.py --dataset ieee_cis --model logreg` | IEEE-CIS | `leakage_comparison_ieee_cis.csv/.png` |
| 2026-11-25 | `run_drift_analysis.py --dataset ieee_cis --model logreg` | IEEE-CIS | `psi/ks/drift_events/degradation_ieee_cis.csv`, `degradation_ieee_cis.png` |
| 2026-11-25 | `run_retraining_experiments.py --dataset ieee_cis --model logreg` | IEEE-CIS | `retraining_metrics/costs_ieee_cis.csv`, `retraining_pr_auc_ieee_cis.png` |

All runs are reproducible with seed 42. Role 3's implementation was replaced by
the owning teammate's branch (`7691991`) and the full suite then passed 80 tests.

## Results — Objective 1: leakage inflation

| Dataset | Leaky PR-AUC | Leakage-free PR-AUC | Leaky recall@1%FPR | Clean recall@1%FPR |
|---|---|---|---|---|
| Sparkov (21 folds) | **0.2275** | **0.0061** (~37× inflation) | 0.5611 | 0.0328 |
| IEEE-CIS (4 folds) | **0.4321** | **0.2519** (~1.7× inflation) | 0.3564 | 0.2478 |

Leaky rows = random split + encoders/preprocessing fitted on all data.
Leakage-free rows = chronological walk-forward with train-only fitting.
The inflation replicates on both datasets; its size depends on the dataset and
on how strongly the encoded features correlate with the label.

## Results — Objective 2: degradation

- Sparkov: static PR-AUC **0.0105 → 0.0045** over 21 months (peak 0.0145);
  PSI max 0.00056 → concept drift, not marginal feature shift.
- IEEE-CIS: static PR-AUC **0.4236 → 0.1740** over the first 3 test months,
  recovering to 0.3944 in the final month (only 2,754 rows — noisy).
  PSI max 0.22 (`dist2`); drift events: ADWIN 89, Page-Hinkley 72, DDM 11.
- Caveat: drift-trigger alerts are sensitive to the operating threshold.

## Results — Objective 3: retraining strategies

| Strategy | Sparkov PR-AUC / recall@FPR / cost | IEEE PR-AUC / recall@FPR / cost |
|---|---|---|
| Sliding window | **0.0116** / 0.0149 / **4305** | **0.3797** / 0.2707 / **19800** |
| Static | 0.0102 / 0.0103 / 4361 | 0.3444 / 0.2383 / 19975 |
| Recency-weighted | 0.0083 / **0.0368** / 4410 | 0.3794 / **0.2723** / 19875 |
| Periodic | 0.0064 / 0.0308 / 4368 | 0.3715 / 0.2704 / 19865 |
| Drift-triggered | 0.0064 / 0.0308 / 4368 | 0.3715 / 0.2704 / 19865 |

Training time (Sparkov / IEEE total): static 2 s / 106 s; sliding 63 s / 355 s;
periodic 351 s / 1368 s; weighted 323 s / 766 s; drift-triggered 348 s / 712 s.
Sliding window is the best PR-AUC/cost trade-off on both datasets.

## Decisions log

| Decision | Value | Status |
|---|---|---|
| IEEE reference date | 2017-12-01 | assumed; gives 7 months |
| Split | 3 train / 1 test month, expanding | from config |
| Embargo (`gap_days`) | 0 | assumed — real chargeback lag not modelled |
| FPR target / cost ratio | 0.01 / 10:1 | assumed |
| One-hot min frequency | 100 | memory cap for real data |
| Model for reported runs | Logistic Regression | 8 GB RAM constraint |

## Open questions / TODOs

1. IEEE has only 7 months and a partial last month; more history would give more folds.
2. Re-run with XGBoost/Random Forest on sampled data if time allows.
3. Calibrate drift-trigger thresholds against alert precision rather than a 0.5 rule.
4. Decide `gap_days` from the real chargeback lag; feedback currently assumes immediate labels.
5. Fill `docs/literature_review.md` — still an empty template.

## Leakage checklist for reported results

- [x] Chronological month splits with configured embargo
- [x] No random split outside `run_leaky_baseline()`
- [x] All learned state fitted on train fold only
- [x] Expanding past-only training encodings; frozen mapping on test
- [x] Past-only aggregates use `closed="left"` windows
- [x] Metrics include PR-AUC, recall@FPR, F1 and base rate
- [x] Every table traces to a run log entry above
