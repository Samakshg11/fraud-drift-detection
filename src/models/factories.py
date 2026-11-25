"""REFERENCE IMPLEMENTATION for Role 4 (Modeling) -- safe to replace.

Model factories with explicit, train-fold-only imbalance handling.

For XGBoost, ``scale_pos_weight`` must be computed from the training fold's
labels, never from the full dataset. ``xgb_pos_weight(y_train)`` does exactly
that; call it inside the fold loop and pass the result to ``build_model``.
Random Forest and Logistic Regression use ``class_weight="balanced"``, which
scikit-learn derives from the labels passed to ``fit`` (also train-only).
"""

from __future__ import annotations

import numpy as np

MODEL_NAMES = ("logreg", "rf", "xgboost")


def imbalance_ratio(y_train: np.ndarray) -> float:
    """negative-count / positive-count for a training fold."""
    y = np.asarray(y_train, dtype=int)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        raise ValueError("No positive labels in training fold; cannot fit a fraud model")
    return float(negatives / positives) if negatives else 1.0


def xgb_pos_weight(y_train: np.ndarray) -> float:
    return imbalance_ratio(y_train)


def build_model(name: str, *, seed: int, scale_pos_weight: float = 1.0):
    """Return an unfitted estimator for ``name``.

    Parameters
    ----------
    name:
        One of ``MODEL_NAMES``.
    seed:
        Reproducibility seed (config value).
    scale_pos_weight:
        Only used by XGBoost. Compute it from the training fold with
        :func:`xgb_pos_weight`.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    # Lazy import: XGBoost needs libomp on macOS and is not needed for other models.
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=seed,
    )
