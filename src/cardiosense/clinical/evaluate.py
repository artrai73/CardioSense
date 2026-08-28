"""Evaluation, model selection and error analysis for the clinical pipeline.

The discipline this module enforces:

* **Model selection happens on validation, never on test.** :func:`select_model`
  takes validation metrics only. The test split is scored once, at the end.
* **The threshold is tuned on validation too**, and metrics are reported at both
  the tuned threshold and the default 0.5 so the effect of tuning is visible.
* **Selection is not by accuracy.** ROC-AUC ranks the models; a near-tie is
  broken in favour of the simpler model, because on ~45 test patients a 0.005 AUC
  difference is indistinguishable from noise and the simpler model is easier to
  explain and to calibrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..common.config import Config
from ..common.io_utils import save_dataframe, save_json
from ..common.logging_utils import get_logger
from ..common.metrics import binary_metrics, bootstrap_ci, find_best_threshold
from ..common.paths import ensure_dir
from ..common.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve

__all__ = [
    "predict_proba",
    "evaluate_at_threshold",
    "tune_threshold",
    "out_of_fold_probabilities",
    "build_comparison_table",
    "select_model",
    "export_errors",
    "evaluate_final",
]

logger = get_logger(__name__)

#: Simplicity ranking used to break near-ties. Lower is simpler.
_COMPLEXITY_ORDER = {"logistic_regression": 0, "xgboost": 1}


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(positive class) as a 1-D array.

    Works for any estimator with ``predict_proba``; falls back to
    ``decision_function`` squashed through a logistic, which keeps the ranking
    intact for AUC even if the values are not true probabilities.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        logger.warning("%s has no predict_proba; using a squashed decision_function. "
                       "These values are scores, not calibrated probabilities.",
                       type(model).__name__)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError(f"{type(model).__name__} exposes neither predict_proba nor "
                         "decision_function.")


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full binary metric block at a given decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    metrics = binary_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    return metrics


def out_of_fold_probabilities(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    folds: int = 5,
) -> np.ndarray:
    """Cross-validated out-of-fold probabilities on the training split.

    ``cross_val_predict`` refits a clone of the estimator on each set of k-1 folds
    and predicts the held-out fold, so no row is scored by a model that saw it.
    These predictions are legitimate for threshold selection: they come from
    models trained the same way on the same data distribution.

    The caveat, stated plainly: they come from k *different* models, not from the
    final one. They estimate the threshold the training procedure produces, not
    the exact final fit. On ~200 patients that trade is worth taking — the
    alternative is tuning on 45 validation points, where the chosen threshold
    swings wildly with a single flipped case.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg.seed))
    proba = cross_val_predict(estimator, X, y, cv=splitter, method="predict_proba", n_jobs=-1)
    return np.asarray(proba)[:, 1]


def tune_threshold(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    cfg: Config,
    oof_y: np.ndarray | None = None,
    oof_prob: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Select the operating threshold WITHOUT touching the test split.

    Strategy comes from ``selection.threshold_strategy``:

    * ``youden``  — maximise sensitivity + specificity - 1. The standard choice in
      diagnostic medicine: it weights missing a diseased patient and alarming a
      healthy one equally, and it does not chase precision, which is unstable on
      a small sample.
    * ``f1``      — maximise F1.
    * ``fixed``   — use ``selection.fixed_threshold`` unchanged.

    Which data the threshold is fitted on comes from
    ``selection.threshold_tuning_data``:

    * ``cv_oof_plus_val`` (default) — out-of-fold training predictions pooled with
      the validation predictions. **This matters.** Tuning on 45 validation
      patients alone produces a threshold that is itself a high-variance estimate:
      in testing, a validation-only Youden threshold of 0.65 collapsed test recall
      to 0.09, because the threshold had fit the validation noise rather than the
      operating point. Pooling gives ~250 points and a far more stable cut.
    * ``validation`` — validation only. Kept as an option so the difference can be
      reported as an ablation.

    Args:
        y_val: Validation labels.
        y_val_prob: Validation probabilities.
        cfg: Clinical configuration.
        oof_y: Training labels, for the pooled strategy.
        oof_prob: Out-of-fold training probabilities, from
            :func:`out_of_fold_probabilities`.

    Returns:
        ``(threshold, info_dict)``.
    """
    strategy = str(cfg.selection.get("threshold_strategy", "youden")).lower()
    fixed = float(cfg.selection.get("fixed_threshold", 0.5))

    if strategy == "fixed":
        return fixed, {"strategy": "fixed", "threshold": fixed, "tuned_on": "none"}

    metric = "youden" if strategy == "youden" else "f1"
    source = str(cfg.selection.get("threshold_tuning_data", "cv_oof_plus_val")).lower()

    y_val = np.asarray(y_val).ravel()
    y_val_prob = np.asarray(y_val_prob).ravel()

    use_pooled = source == "cv_oof_plus_val" and oof_prob is not None and oof_y is not None
    if use_pooled:
        y_tune = np.concatenate([np.asarray(oof_y).ravel(), y_val])
        p_tune = np.concatenate([np.asarray(oof_prob).ravel(), y_val_prob])
        tuned_on = "out-of-fold train predictions + validation"
    else:
        y_tune, p_tune = y_val, y_val_prob
        tuned_on = "validation"
        if source == "cv_oof_plus_val":
            logger.warning("Pooled threshold tuning requested but no out-of-fold predictions "
                           "were supplied; falling back to validation only.")

    threshold, score = find_best_threshold(y_tune, p_tune, metric=metric)

    # Report what the validation-only choice would have been, so the difference
    # is visible in the results rather than hidden inside a config flag.
    val_only_threshold, val_only_score = find_best_threshold(y_val, y_val_prob, metric=metric)

    info = {
        "strategy": strategy,
        "threshold": round(float(threshold), 4),
        "objective": metric,
        "objective_value": round(float(score), 4),
        "tuned_on": tuned_on,
        "n_tuning_samples": int(len(y_tune)),
        "validation_only_alternative": {
            "threshold": round(float(val_only_threshold), 4),
            "objective_value": round(float(val_only_score), 4),
            "n_samples": int(len(y_val)),
        },
        "note": "Tuned without touching the test split. Test metrics are reported at this "
                "threshold AND at 0.5, so the effect of tuning is visible.",
    }
    logger.info("Threshold tuned on %s (%s, n=%d): %.3f | validation-only would give %.3f",
                tuned_on, metric, len(y_tune), threshold, val_only_threshold)
    return float(threshold), info


def build_comparison_table(
    results: Mapping[str, Mapping[str, Any]],
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assemble the model comparison table.

    Args:
        results: ``{model_name: {"val": {...}, "cv": {...}, ...}}``.
        metrics: Metric names to include, in order.

    Returns:
        A DataFrame, one row per model.
    """
    metric_names = list(metrics or ["accuracy", "precision", "recall", "f1",
                                    "roc_auc", "pr_auc", "brier"])
    rows = []
    for name, blocks in results.items():
        val = blocks.get("val", {})
        row: dict[str, Any] = {"model": name}
        row["cv_roc_auc"] = round(float(blocks.get("cv_score", float("nan"))), 4) \
            if blocks.get("cv_score") is not None else np.nan
        for metric in metric_names:
            value = val.get(metric)
            row[f"val_{metric}"] = round(float(value), 4) if value is not None else np.nan
        row["fit_seconds"] = blocks.get("search_seconds", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(
    results: Mapping[str, Mapping[str, Any]],
    cfg: Config,
) -> tuple[str, dict[str, Any]]:
    """Choose the final model using VALIDATION metrics.

    Ranking is by ``selection.primary_metric`` (ROC-AUC by default), not accuracy.
    Accuracy depends on an arbitrary threshold and, on a small validation set,
    moves in 2-point jumps; ROC-AUC is threshold-independent and uses the full
    ranking of the predictions.

    If the best two models are within ``selection.tie_tolerance``, the simpler one
    wins. The justification is explicit: at n_val ~ 45 a gap of 0.01 AUC is well
    inside the sampling noise, and a Logistic Regression is easier to explain to a
    clinician, cheaper to calibrate reliably, and less likely to have latched onto
    a spurious interaction.

    Args:
        results: ``{model_name: {"val": {...}}}``.
        cfg: Clinical configuration.

    Returns:
        ``(selected_name, decision_record)``.
    """
    metric = str(cfg.selection.get("primary_metric", "roc_auc"))
    tolerance = float(cfg.selection.get("tie_tolerance", 0.0))

    scored = {
        name: float(blocks.get("val", {}).get(metric, float("nan")))
        for name, blocks in results.items()
    }
    valid = {k: v for k, v in scored.items() if not np.isnan(v)}
    if not valid:
        raise ValueError(f"No model produced a validation {metric}; cannot select.")

    ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]

    decision: dict[str, Any] = {
        "primary_metric": metric,
        "evaluated_on": "validation",
        "scores": {k: round(v, 4) for k, v in scored.items()},
        "ranked": [name for name, _ in ranked],
        "tie_tolerance": tolerance,
        "tie_broken": False,
    }

    contenders = [name for name, score in ranked if best_score - score <= tolerance]
    if len(contenders) > 1:
        simplest = min(contenders, key=lambda n: _COMPLEXITY_ORDER.get(n, 99))
        if simplest != best_name:
            decision["tie_broken"] = True
            decision["tie_reason"] = (
                f"{best_name} led {simplest} by {best_score - scored[simplest]:.4f} {metric}, "
                f"within the tie tolerance of {tolerance}. On a validation split this small "
                f"that gap is not distinguishable from sampling noise, so the simpler model "
                f"was selected: it is easier to explain, calibrates more reliably, and is "
                f"less likely to have fit a spurious interaction."
            )
            logger.info("Tie broken in favour of the simpler model: %s over %s",
                        simplest, best_name)
            best_name = simplest

    decision["selected"] = best_name
    decision["selected_score"] = round(scored[best_name], 4)
    logger.info("Selected model: %s (validation %s = %.4f)",
                best_name, metric, scored[best_name])
    return best_name, decision


def export_errors(
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_dir: Path | str,
    prefix: str = "test",
    n_examples: int = 10,
) -> dict[str, Any]:
    """Export misclassified patients for error analysis.

    Writes:

    * ``<prefix>_errors.csv`` — every misclassified patient with their raw
      (un-transformed) feature values, predicted probability and error type.
    * ``<prefix>_predictions.csv`` — every patient, for downstream analysis.

    Rows are sorted by how confidently wrong the model was. The most useful cases
    are the confident errors: a false negative at p = 0.05 is a patient the model
    was sure was healthy, which is the failure mode that matters clinically.

    Args:
        X_raw: Raw feature frame for the split (indices aligned with y).
        y_true: True labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        out_dir: Output directory.
        prefix: Filename prefix (``"test"``, ``"val"``).
        n_examples: How many of each error type to report in the summary.

    Returns:
        Summary dict with counts and the most confident errors.
    """
    out = ensure_dir(out_dir)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    frame = X_raw.copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["probability"] = np.round(y_prob, 4)
    frame["margin"] = np.round(np.abs(y_prob - threshold), 4)
    frame["correct"] = y_true == y_pred
    frame["error_type"] = np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 0),
         (y_true == 0) & (y_pred == 1), (y_true == 1) & (y_pred == 0)],
        ["TP", "TN", "FP", "FN"], default="?",
    )

    save_dataframe(frame.reset_index().rename(columns={"index": "patient_index"}),
                   out / f"{prefix}_predictions.csv")

    errors = frame[~frame["correct"]].sort_values("margin", ascending=False)
    save_dataframe(errors.reset_index().rename(columns={"index": "patient_index"}),
                   out / f"{prefix}_errors.csv")

    false_negatives = errors[errors.error_type == "FN"]
    false_positives = errors[errors.error_type == "FP"]

    summary: dict[str, Any] = {
        "split": prefix,
        "threshold": float(threshold),
        "n_total": int(len(frame)),
        "n_errors": int(len(errors)),
        "error_rate": round(float(len(errors) / max(len(frame), 1)), 4),
        "counts": {k: int(v) for k, v in frame.error_type.value_counts().items()},
        "mean_confidence_when_correct": round(
            float(frame.loc[frame.correct, "margin"].mean()), 4) if frame.correct.any() else None,
        "mean_confidence_when_wrong": round(
            float(errors["margin"].mean()), 4) if len(errors) else None,
        "most_confident_false_negatives": false_negatives.head(n_examples)[
            ["y_true", "probability", "margin"]].reset_index().to_dict(orient="records"),
        "most_confident_false_positives": false_positives.head(n_examples)[
            ["y_true", "probability", "margin"]].reset_index().to_dict(orient="records"),
        "files": {
            "errors": str((out / f"{prefix}_errors.csv").name),
            "predictions": str((out / f"{prefix}_predictions.csv").name),
        },
    }

    logger.info("[%s] %d errors of %d (%.1f%%): %d FN, %d FP",
                prefix, len(errors), len(frame), 100 * summary["error_rate"],
                len(false_negatives), len(false_positives))

    if summary["mean_confidence_when_wrong"] is not None and \
            summary["mean_confidence_when_correct"] is not None and \
            summary["mean_confidence_when_wrong"] >= summary["mean_confidence_when_correct"]:
        logger.warning(
            "The model is on average as confident when wrong as when right. "
            "That is a calibration problem, not an accuracy problem — see calibrate.py.",
        )

    return summary


def evaluate_final(
    model_name: str,
    y_test: np.ndarray,
    y_test_prob: np.ndarray,
    threshold: float,
    cfg: Config,
    out_dir: Path | str,
    extra_curves: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Score the selected model on the TEST split and write all final figures.

    This is the only place the test split is scored. It produces metrics at the
    tuned threshold and at 0.5, bootstrap confidence intervals for ROC-AUC and
    PR-AUC, a confusion matrix, and ROC / PR curves.

    Args:
        model_name: Name of the selected model, used in figure titles.
        y_test: True test labels.
        y_test_prob: Predicted probabilities on test.
        threshold: Threshold chosen on validation.
        cfg: Clinical configuration.
        out_dir: ``results/clinical``.
        extra_curves: Additional ``{label: (y_true, y_score)}`` overlays, e.g. the
            rejected model, so the report can show both curves on one axis.

    Returns:
        The full test metric block.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = ensure_dir(out_dir)
    y_test = np.asarray(y_test).ravel().astype(int)
    y_test_prob = np.asarray(y_test_prob).ravel()

    tuned = evaluate_at_threshold(y_test, y_test_prob, threshold)
    default = evaluate_at_threshold(y_test, y_test_prob, 0.5)

    n_boot = int(cfg.get("evaluation.n_bootstrap", 1000) or 1000)
    intervals = {
        "roc_auc": bootstrap_ci(y_test, y_test_prob, roc_auc_score,
                                n_bootstrap=n_boot, seed=int(cfg.seed)),
        "pr_auc": bootstrap_ci(y_test, y_test_prob, average_precision_score,
                               n_bootstrap=n_boot, seed=int(cfg.seed)),
    }

    curves = {model_name: (y_test, y_test_prob)}
    if extra_curves:
        curves.update(extra_curves)

    plot_confusion_matrix(
        y_test, (y_test_prob >= threshold).astype(int), out / "confusion_matrix.png",
        class_names=["No disease", "Disease"],
        title=f"{model_name} — test confusion matrix (threshold = {threshold:.2f})",
    )
    plot_confusion_matrix(
        y_test, (y_test_prob >= threshold).astype(int), out / "confusion_matrix_normalized.png",
        class_names=["No disease", "Disease"], normalize="true",
        title=f"{model_name} — test confusion matrix (row-normalised)",
    )
    plot_roc_curve(curves, out / "roc_curve.png", title="ROC — test split")
    plot_pr_curve(curves, out / "pr_curve.png", title="Precision-Recall — test split")

    block = {
        "model": model_name,
        "split": "test",
        "n_samples": int(len(y_test)),
        "positive_rate": round(float(y_test.mean()), 4),
        "at_tuned_threshold": tuned,
        "at_default_threshold_0.5": default,
        "confidence_intervals_95pct": intervals,
    }

    logger.info(
        "TEST — %s: ROC-AUC %.3f [%.3f, %.3f] | recall %.3f | precision %.3f | F1 %.3f | "
        "Brier %.3f",
        model_name, intervals["roc_auc"]["point"], intervals["roc_auc"]["lower"],
        intervals["roc_auc"]["upper"], tuned["recall"], tuned["precision"], tuned["f1"],
        tuned["brier"],
    )
    logger.info("Confidence interval width is %.3f AUC — that is the honest precision of this "
                "estimate on %d patients.",
                intervals["roc_auc"]["upper"] - intervals["roc_auc"]["lower"], len(y_test))

    save_json(block, out / "test_metrics.json")
    return block
