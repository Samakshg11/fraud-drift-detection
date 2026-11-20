# Leakage-Free and Drift-Aware Credit Card Fraud Detection

Leakage-free ML pipeline for credit card fraud detection with time-based splits on long-horizon datasets (IEEE-CIS, Sparkov). We measure how models degrade over months and compare drift-aware retraining strategies (sliding window, drift-triggered updates) using XGBoost and Random Forest.

## Setup
```bash
git clone https://github.com/Samakshg11/fraud-drift-detection.git
cd fraud-drift-detection
git checkout dev
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets (not stored in repo)
Download into `data/raw/`:
- IEEE-CIS Fraud Detection (Kaggle)
- Sparkov simulated credit card fraud (Kaggle)

## Folder structure
- `src/` : all code (data, pipeline, features, models, drift, retraining, evaluation)
- `notebooks/` : experiments and EDA
- `tests/` : leakage tests
- `results/` : figures and tables
- `paper/` : report/paper sections
- `docs/` : task assignment, literature review

## Git rules
- No direct push to `main` or `dev`. Make a branch `feature/<name>` and open a Pull Request into `dev`.
- At least 1 teammate reviews before merge.
- Never commit data or model files.
