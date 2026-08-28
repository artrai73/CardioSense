"""Classical baseline for the ECG task (Experiment E-A).

Hand-crafted per-lead statistics fed to one-vs-rest Logistic Regression. Its job
is not to be good; its job is to establish the floor that the CNN must clear
before anyone claims deep learning was necessary. A CNN that beats chance but not
this baseline has learned nothing a summary statistic could not.

Features, per lead (11 x 12 = 132 total)
----------------------------------------

======================  ====================================================
Feature                 What it captures
======================  ====================================================
``mean``                Residual DC offset after high-pass filtering
``std``                 Overall signal amplitude / variability
``min``, ``max``        Deflection extremes (Q depth, R height)
``ptp``                 Peak-to-peak range, an amplitude proxy relevant to HYP
``rms``                 Signal power
``skew``                Waveform asymmetry — QRS complexes are sharply asymmetric
``kurtosis``            "Peakedness"; high values mean sharp spikes over a flat
                        baseline, which is what a normal QRS looks like
``zero_crossing_rate``  A crude frequency proxy; elevated in conduction
                        disturbances with wide, fragmented complexes
``line_length``         Sum of |consecutive differences| — total waveform
                        tortuosity, a standard feature in seizure and
                        arrhythmia detection
``energy``              Sum of squares, a duplicate-ish power measure kept
                        because it scales differently from RMS
======================  ====================================================

What this baseline structurally cannot do: these are all *global* statistics over
10 seconds. They discard the temporal ordering entirely, so nothing about where a
deflection sits relative to the QRS survives. ST-segment elevation and depression
are defined by exactly that relationship, which predicts that STTC and MI should
be the classes where the CNN gains most. Worth checking against the actual
per-class results rather than assuming.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

from ..common.config import Config
from ..common.logging_utils import get_logger

__all__ = ["FEATURE_FUNCTIONS", "extract_features", "extract_feature_matrix",
           "feature_names", "train_baseline"]

logger = get_logger("ecg.baseline")

FEATURE_FUNCTIONS: dict[str, Any] = {
    "mean": lambda x: x.mean(axis=-1),
    "std": lambda x: x.std(axis=-1),
    "min": lambda x: x.min(axis=-1),
    "max": lambda x: x.max(axis=-1),
    "ptp": lambda x: x.max(axis=-1) - x.min(axis=-1),
    "rms": lambda x: np.sqrt((x ** 2).mean(axis=-1)),
    "skew": lambda x: stats.skew(x, axis=-1),
    "kurtosis": lambda x: stats.kurtosis(x, axis=-1),
    "zero_crossing_rate": lambda x: (np.diff(np.signbit(x), axis=-1) != 0).mean(axis=-1),
    "line_length": lambda x: np.abs(np.diff(x, axis=-1)).sum(axis=-1),
    "energy": lambda x: (x ** 2).sum(axis=-1),
}


def feature_names(features: Sequence[str], lead_names: Sequence[str]) -> list[str]:
    """Column names for the feature matrix, ordered ``lead_feature``."""
    return [f"{lead}_{feature}" for lead in lead_names for feature in features]


def extract_features(waveform: np.ndarray, features: Sequence[str]) -> np.ndarray:
    """Extract statistics from one record.

    Args:
        waveform: Shape ``(n_leads, n_samples)``.
        features: Names from :data:`FEATURE_FUNCTIONS`.

    Returns:
        A flat vector of length ``n_leads * len(features)``, lead-major.
    """
    x = np.asarray(waveform, dtype=np.float64)
    columns = []
    for name in features:
        if name not in FEATURE_FUNCTIONS:
            raise KeyError(f"Unknown feature {name!r}. Available: {sorted(FEATURE_FUNCTIONS)}")
        columns.append(np.asarray(FEATURE_FUNCTIONS[name](x), dtype=np.float64))
    # Stack as (n_leads, n_features) then flatten so ordering matches feature_names.
    return np.nan_to_num(np.stack(columns, axis=-1).ravel(), nan=0.0, posinf=0.0, neginf=0.0)


def extract_feature_matrix(
    waveforms: np.ndarray,
    indices: Sequence[int] | np.ndarray,
    features: Sequence[str],
    show_progress: bool = True,
) -> np.ndarray:
    """Build the feature matrix for a split.

    Args:
        waveforms: Full (possibly memory-mapped) waveform array.
        indices: Rows belonging to this split.
        features: Feature names.
        show_progress: Show a progress bar.

    Returns:
        Shape ``(len(indices), n_leads * len(features))``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    iterator: Any = indices
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(indices, desc="extracting features")

    rows = [extract_features(np.asarray(waveforms[int(i)]), features) for i in iterator]
    return np.vstack(rows).astype(np.float32)


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    cfg: Config,
    X_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one-vs-rest Logistic Regression on the statistical features.

    One independent binary classifier per superclass, which is the correct
    formulation for a multi-label target: each class is predicted on its own, and
    a record can be positive for several.

    Standardisation is fitted on **train only** and reused for val/test, the same
    discipline as the clinical pipeline.

    Args:
        X_train: Training feature matrix.
        y_train: Multi-hot training labels.
        X_val: Validation feature matrix.
        cfg: ECG configuration.
        X_test: Optional test feature matrix.

    Returns:
        Dict with the fitted models, the scaler, and predicted probabilities.
    """
    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ..common.compat import logistic_penalty_kwargs

    params = cfg.baseline
    classes = list(cfg.task.classes)

    estimator = LogisticRegression(
        max_iter=int(params.get("max_iter", 1000)),
        class_weight=params.get("class_weight", "balanced"),
        random_state=int(cfg.seed),
        **logistic_penalty_kwargs("l2"),
    )
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", MultiOutputClassifier(estimator, n_jobs=-1)),
    ])

    logger.info("Fitting one-vs-rest baseline: %d features, %d records, %d classes",
                X_train.shape[1], X_train.shape[0], len(classes))
    start = time.time()
    pipeline.fit(X_train, y_train.astype(int))
    elapsed = time.time() - start

    def probabilities(X: np.ndarray) -> np.ndarray:
        # MultiOutputClassifier returns a list of (n, 2) arrays, one per class.
        per_class = pipeline.predict_proba(X)
        return np.column_stack([p[:, 1] for p in per_class]).astype(np.float32)

    result: dict[str, Any] = {
        "model": pipeline,
        "classes": classes,
        "n_features": int(X_train.shape[1]),
        "train_seconds": round(elapsed, 2),
        "val_prob": probabilities(X_val),
    }
    if X_test is not None:
        result["test_prob"] = probabilities(X_test)

    logger.info("Baseline fitted in %.1fs", elapsed)
    return result
