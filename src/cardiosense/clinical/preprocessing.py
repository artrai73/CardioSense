"""Splitting and preprocessing for the clinical pipeline.

Two responsibilities, both of which are where tabular projects usually leak:

1. :func:`split_data` — a reproducible, stratified train/validation/test split.
2. :func:`build_preprocessor` — a ``ColumnTransformer`` that is **fit on the
   training split only** and then applied unchanged to validation, test and any
   future patient at inference time.

Why the preprocessor is a fitted object and not a function: the median used to
impute ``ca``, the category levels seen by the one-hot encoder, and the mean and
standard deviation used by the scaler are all *learned parameters*. Recomputing
them on the test set — or on the full dataset before splitting — lets information
about the test patients influence the training representation. That inflates
every reported metric and is invisible in the results. Fitting once on train and
pickling the fitted object is what makes the inference script honest.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from ..common.config import Config
from ..common.logging_utils import get_logger

__all__ = ["DataSplits", "split_data", "build_preprocessor", "fit_preprocessor",
           "get_feature_names", "transform_splits"]

logger = get_logger(__name__)


class DataSplits(NamedTuple):
    """The three splits, kept as DataFrames so feature names survive."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    summary: dict[str, Any]


def split_data(X: pd.DataFrame, y: pd.Series, cfg: Config) -> DataSplits:
    """Produce a reproducible, stratified train / validation / test split.

    The split is done in two stages:

    * Stage 1 carves out the **test** set (``split.test_size`` of the full data).
      This set is written down and then left alone until the final evaluation.
    * Stage 2 carves the **validation** set out of what remains. ``split.val_size``
      is expressed as a fraction of the FULL dataset, so ``0.15`` really means 15%
      of all patients, not 15% of the remainder — the second call therefore uses
      an adjusted fraction.

    Both stages stratify on the target, so the ~46% positive rate is preserved in
    every split. With only ~300 patients this matters: an unstratified draw can
    easily produce a test set that is 60% positive, which makes the metrics
    incomparable to the other splits.

    The three splits have distinct, non-overlapping jobs:

    * **train** — fits the preprocessor, the models, and the cross-validated
      hyperparameter search.
    * **validation** — selects between models, tunes the decision threshold, and
      fits the probability calibrator. Never used to fit model weights.
    * **test** — touched exactly once, for the numbers that go in the report.

    Args:
        X: Feature frame.
        y: Binary target.
        cfg: Clinical configuration (uses ``seed`` and the ``split`` block).

    Returns:
        A :class:`DataSplits` with a summary dict describing the result.

    Raises:
        ValueError: If the requested fractions do not leave a usable train split.
    """
    seed = int(cfg.seed)
    test_size = float(cfg.split.test_size)
    val_size = float(cfg.split.val_size)
    stratify = bool(cfg.split.get("stratify", True))

    if not 0 < test_size < 1 or not 0 <= val_size < 1:
        raise ValueError(f"Invalid split sizes: test={test_size}, val={val_size}")
    if test_size + val_size >= 0.9:
        raise ValueError(
            f"test_size + val_size = {test_size + val_size:.2f} leaves too little for training."
        )

    strat_full = y if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        shuffle=bool(cfg.split.get("shuffle", True)),
        stratify=strat_full,
    )

    # val_size is a fraction of the FULL dataset; rescale it for the remainder.
    relative_val = val_size / (1.0 - test_size)
    strat_trainval = y_trainval if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=seed,
        shuffle=True,
        stratify=strat_trainval,
    )

    summary = {
        "seed": seed,
        "stratified": stratify,
        "requested_fractions": {"train": round(1 - test_size - val_size, 4),
                                "val": val_size, "test": test_size},
        "sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test)),
                  "total": int(len(X))},
        "positive_rate": {
            "overall": round(float(y.mean()), 4),
            "train": round(float(y_train.mean()), 4),
            "val": round(float(y_val.mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "class_counts": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "val": {str(k): int(v) for k, v in y_val.value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        # Index overlap must be empty. Asserted below and recorded so the check
        # is visible in the results rather than merely believed.
        "index_overlap": {
            "train_val": len(set(X_train.index) & set(X_val.index)),
            "train_test": len(set(X_train.index) & set(X_test.index)),
            "val_test": len(set(X_val.index) & set(X_test.index)),
        },
    }

    assert sum(summary["index_overlap"].values()) == 0, "Split overlap detected — leakage!"

    logger.info("Split: train=%d, val=%d, test=%d (positive rate %.3f / %.3f / %.3f)",
                len(X_train), len(X_val), len(X_test),
                y_train.mean(), y_val.mean(), y_test.mean())

    if len(X_test) < 40:
        logger.warning(
            "Test split has only %d patients. Report confidence intervals, not point "
            "estimates — a single flipped prediction moves accuracy by %.1f points.",
            len(X_test), 100 / len(X_test),
        )

    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test, summary)


def build_preprocessor(cfg: Config) -> ColumnTransformer:
    """Build the (unfitted) preprocessing ColumnTransformer.

    Two branches:

    * **Numeric** — median imputation, then scaling. Median rather than mean
      because ``chol`` and ``oldpeak`` are right-skewed and the mean would be
      dragged by outliers.
    * **Categorical** — most-frequent imputation, then one-hot encoding with
      ``handle_unknown`` set so that a category never seen during training does
      not crash inference on a new patient.

    Scaling is applied even though XGBoost does not need it. The same fitted
    transformer serves both models and the inference script, and scaling is a
    monotonic per-feature transform, so it cannot change a tree model's splits —
    it only changes the numbers they are compared against. Keeping one
    transformer removes a whole class of "which preprocessor did this model
    expect?" bugs at deployment.

    Args:
        cfg: Clinical configuration.

    Returns:
        An unfitted ``ColumnTransformer``.
    """
    pre = cfg.preprocessing
    numeric = list(cfg.dataset.numeric_features)
    categorical = list(cfg.dataset.categorical_features)

    scaler_name = str(pre.get("scaler", "standard")).lower()
    if scaler_name == "standard":
        scaler: Any = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name in {"none", "null", ""}:
        scaler = "passthrough"
    else:
        raise ValueError(f"Unknown scaler {scaler_name!r}; use standard, minmax or none.")

    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=str(pre.get("numeric_imputer", "median")))),
    ]
    if scaler != "passthrough":
        numeric_steps.append(("scale", scaler))
    numeric_pipeline = Pipeline(numeric_steps)

    encoder_kwargs: dict[str, Any] = {
        "handle_unknown": str(pre.get("handle_unknown", "infrequent_if_exist")),
        "sparse_output": False,
    }
    min_frequency = pre.get("one_hot_min_frequency")
    if min_frequency:
        encoder_kwargs["min_frequency"] = min_frequency

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=str(pre.get("categorical_imputer", "most_frequent")))),
        ("encode", OneHotEncoder(**encoder_kwargs)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> ColumnTransformer:
    """Fit the preprocessor on the TRAINING SPLIT ONLY.

    This function exists so the rule has a name and a single call site. If you
    ever find yourself calling ``fit`` or ``fit_transform`` on validation or test
    data, that is the leak.
    """
    logger.info("Fitting preprocessor on %d training rows (train only — no leakage).",
                len(X_train))
    return preprocessor.fit(X_train, y_train)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output feature names of a fitted ColumnTransformer.

    These are what SHAP labels its axes with, so they must match the transformed
    matrix column-for-column.
    """
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # noqa: BLE001 - very old sklearn fallback
        n_features = getattr(preprocessor, "n_features_in_", 0)
        return [f"feature_{i}" for i in range(n_features)]


def transform_splits(
    preprocessor: ColumnTransformer,
    splits: DataSplits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Apply a FITTED preprocessor to all three splits.

    Returns:
        ``(X_train_t, X_val_t, X_test_t, feature_names)``.
    """
    X_train_t = preprocessor.transform(splits.X_train)
    X_val_t = preprocessor.transform(splits.X_val)
    X_test_t = preprocessor.transform(splits.X_test)
    names = get_feature_names(preprocessor)

    logger.info("Transformed feature matrix: %d raw features -> %d encoded features",
                splits.X_train.shape[1], X_train_t.shape[1])

    if X_train_t.shape[1] != len(names):
        raise RuntimeError(
            f"Feature-name mismatch: matrix has {X_train_t.shape[1]} columns but "
            f"{len(names)} names. SHAP plots would be mislabelled."
        )
    return X_train_t, X_val_t, X_test_t, names
