"""Metric computation shared by all three pipelines.

Design rule: a function only ever returns metrics that are *mathematically
appropriate* for its task type. There is no single ``compute_metrics`` that
quietly reports multi-class accuracy on a multi-label problem.

* :func:`binary_metrics`      — clinical pipeline, X-ray pipeline
* :func:`multilabel_metrics`  — ECG pipeline (diagnostic superclasses)
* :func:`multiclass_metrics`  — provided for completeness / Phase 2

Calibration measures (:func:`brier_score`, :func:`expected_calibration_error`)
live here too because the clinical pipeline and the eventual fusion stage both
need them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "binary_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
    "brier_score",
    "expected_calibration_error",
    "find_best_threshold",
    "bootstrap_ci",
    "metrics_to_frame",
    "specificity_score",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------
def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate. Reported alongside recall for clinical readability."""
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp + _EPS))


def binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Metrics for a binary classification task.

    Args:
        y_true: Ground-truth labels in {0, 1}.
        y_pred: Hard predictions in {0, 1}.
        y_prob: Predicted probability of the positive class. When supplied,
            ROC-AUC, PR-AUC, Brier score, log loss and ECE are added.
        prefix: Prepended to every key, e.g. ``"val_"``.

    Returns:
        Mapping of metric name to value. Probability-dependent metrics are
        omitted (not set to zero) when *y_prob* is ``None``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp + _EPS)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "support_positive": int(y_true.sum()),
        "support_total": int(y_true.size),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        if len(np.unique(y_true)) > 1:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
            out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, _EPS, 1 - _EPS)))
        out["brier"] = float(brier_score_loss(y_true, y_prob))
        out["ece"] = expected_calibration_error(y_true, y_prob)

    return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


# ---------------------------------------------------------------------------
# Multi-class
# ---------------------------------------------------------------------------
def multiclass_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Metrics for a single-label multi-class task."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            out["macro_roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass  # a class missing from y_true makes OvR AUC undefined

    if class_names is not None:
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0,
                                labels=list(range(len(class_names))))
        out["per_class_f1"] = {name: float(v) for name, v in zip(class_names, per_class_f1)}

    return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


# ---------------------------------------------------------------------------
# Multi-label (ECG)
# ---------------------------------------------------------------------------
def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: float | Sequence[float] = 0.5,
    class_names: Sequence[str] | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Metrics for a multi-label task such as PTB-XL diagnostic superclasses.

    Args:
        y_true: Binary indicator matrix, shape ``(n_samples, n_classes)``.
        y_prob: Predicted probabilities, same shape.
        thresholds: A single threshold or one per class.
        class_names: Names used as keys in the per-class breakdown.
        prefix: Prepended to every key.

    Returns:
        Macro/micro AUC and average precision, macro/micro/weighted F1, Hamming
        loss, exact-match ratio, and a per-class breakdown.

    Note:
        ``exact_match`` is the multi-label analogue of accuracy and is reported
        under that explicit name. Plain ``accuracy`` is deliberately absent —
        it is not well defined for multi-label outputs.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

    n_classes = y_true.shape[1]
    names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(n_classes)]

    thr = np.full(n_classes, float(thresholds)) if np.isscalar(thresholds) \
        else np.asarray(thresholds, dtype=float)
    if thr.shape[0] != n_classes:
        raise ValueError(f"Expected {n_classes} thresholds, got {thr.shape[0]}")

    y_pred = (y_prob >= thr[None, :]).astype(int)

    per_class: dict[str, dict[str, float]] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for idx, name in enumerate(names):
        col_true, col_prob, col_pred = y_true[:, idx], y_prob[:, idx], y_pred[:, idx]
        entry: dict[str, float] = {
            "support": int(col_true.sum()),
            "prevalence": float(col_true.mean()),
            "precision": float(precision_score(col_true, col_pred, zero_division=0)),
            "recall": float(recall_score(col_true, col_pred, zero_division=0)),
            "f1": float(f1_score(col_true, col_pred, zero_division=0)),
            "threshold": float(thr[idx]),
        }
        if len(np.unique(col_true)) > 1:
            entry["roc_auc"] = float(roc_auc_score(col_true, col_prob))
            entry["pr_auc"] = float(average_precision_score(col_true, col_prob))
            aucs.append(entry["roc_auc"])
            aps.append(entry["pr_auc"])
        per_class[name] = entry

    out: dict[str, Any] = {
        "macro_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_pr_auc": float(np.mean(aps)) if aps else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "n_samples": int(y_true.shape[0]),
        "per_class": per_class,
    }

    try:
        out["micro_roc_auc"] = float(roc_auc_score(y_true, y_prob, average="micro"))
    except ValueError:
        pass

    return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome.

    Lower is better. It is a *proper scoring rule*: it rewards a model only when
    the stated probability matches the observed frequency, so it captures both
    discrimination and calibration.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error (ECE).

    Predictions are binned by confidence; ECE is the support-weighted mean
    absolute gap between mean predicted probability and observed frequency
    within each bin. Zero means perfectly calibrated.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probability of the positive class.
        n_bins: Number of bins.
        strategy: ``"uniform"`` (equal-width) or ``"quantile"`` (equal-count).
            Quantile binning is more stable on small validation sets.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    if edges.size < 2:
        return 0.0

    ece = 0.0
    total = y_prob.size
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lower) & (y_prob <= upper)
        if lower == edges[0]:
            mask |= y_prob == lower
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Threshold selection & uncertainty
# ---------------------------------------------------------------------------
def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_steps: int = 200,
) -> tuple[float, float]:
    """Grid-search the decision threshold that maximises *metric*.

    IMPORTANT: only ever call this on the VALIDATION split. Tuning the threshold
    on test data is a form of leakage that inflates the reported F1.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        metric: ``"f1"``, ``"youden"`` (recall + specificity - 1), or
            ``"balanced_accuracy"``.
        n_steps: Number of candidate thresholds between 0 and 1.

    Returns:
        ``(best_threshold, best_score)``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    best_threshold, best_score = 0.5, -np.inf
    for threshold in np.linspace(0.01, 0.99, n_steps):
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            score = recall_score(y_true, y_pred, zero_division=0) + \
                specificity_score(y_true, y_pred) - 1.0
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown threshold metric: {metric!r}")
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Any = roc_auc_score,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a score-based metric.

    Reporting a CI rather than a bare point estimate matters here: the clinical
    test split has only ~45 patients, where an AUC can move by 0.1 on resampling.

    Returns:
        ``{"point": ..., "lower": ..., "upper": ..., "std": ...}``.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = y_true.size

    point = float(metric_fn(y_true, y_score))
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(float(metric_fn(y_true[idx], y_score[idx])))
        except ValueError:
            continue

    if not scores:
        return {"point": point, "lower": float("nan"), "upper": float("nan"), "std": float("nan")}

    arr = np.asarray(scores)
    return {
        "point": point,
        "lower": float(np.percentile(arr, 100 * alpha / 2)),
        "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "std": float(arr.std(ddof=1)),
    }


def metrics_to_frame(results: Mapping[str, Mapping[str, Any]], columns: Sequence[str] | None = None):
    """Turn ``{model_name: metrics_dict}`` into a tidy comparison DataFrame."""
    import pandas as pd

    rows = []
    for model_name, metrics in results.items():
        row: dict[str, Any] = {"model": model_name}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if columns:
        keep = ["model"] + [c for c in columns if c in frame.columns]
        frame = frame[keep]
    return frame
