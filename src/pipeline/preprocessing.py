"""Train-only preprocessing built as a single sklearn estimator.

The returned :class:`ColumnTransformer` must be fitted on the training fold
only and then applied to the test fold. Because imputer statistics, scaler
moments and the one-hot vocabulary are all learned state, this module never
fits anything itself — it only *builds* the object whose ``fit`` the caller
controls.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    scale_numeric: bool = False,
    numeric_impute: str = "median",
    categorical_impute: str = "most_frequent",
    add_missing_indicator: bool = False,
    one_hot_sparse: bool = True,
    min_frequency: int | float | None = None,
) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` for numeric and categorical features.

    Parameters
    ----------
    numeric_cols, categorical_cols:
        Column names handled by each branch. Everything else is dropped.
    scale_numeric:
        Enable ``StandardScaler``. Needed for Logistic Regression, unnecessary
        (and wasteful) for tree models.
    numeric_impute, categorical_impute:
        Imputation strategies passed to ``SimpleImputer``.
    add_missing_indicator:
        Add a binary column marking imputed numeric values. Useful when
        missingness itself is predictive (common for IEEE-CIS ``D*`` columns).
    one_hot_sparse:
        Keep one-hot output sparse to control memory on high-cardinality
        categorical columns.
    min_frequency:
        Group rarer categories into an "infrequent" bucket (sklearn feature).

    Returns
    -------
    ColumnTransformer
        Unfitted estimator. Call ``.fit(X_train)`` then ``.transform(X_test)``.
    """
    numeric_steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(strategy=numeric_impute, add_indicator=add_missing_indicator),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=categorical_impute)),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=one_hot_sparse,
                min_frequency=min_frequency,
            ),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(numeric_cols)),
            ("cat", Pipeline(categorical_steps), list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def infer_feature_columns(
    df: pd.DataFrame, *, exclude: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split DataFrame columns into numeric and categorical feature lists.

    ``exclude`` should contain the target, timestamp, month index and ID
    columns — none of these may be fed to the preprocessor as plain features.

    Returns
    -------
    (numeric_cols, categorical_cols)
    """
    excluded = set(exclude)
    candidates = [c for c in df.columns if c not in excluded]
    numeric = df[candidates].select_dtypes(include=["number"]).columns.tolist()
    categorical = df[candidates].select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    return numeric, categorical
