"""Evaluation for the chest X-ray task.

The governing fact: **cardiomegaly prevalence is low**, so accuracy is not a
useful metric and is never the headline. A model that predicts "negative" for
every image scores about 97.5% accuracy on the full dataset while finding nothing.

Metric hierarchy used throughout:

1. **PR-AUC** (headline). Precision and recall both ignore true negatives, so
   PR-AUC measures performance on the class we actually care about. Its chance
   level is the positive prevalence, which is always reported alongside it — a
   PR-AUC of 0.30 at 2.5% prevalence is a twelvefold improvement over chance, and
   without the prevalence beside it that number looks like failure.
2. **ROC-AUC**, for comparability with published ChestX-ray14 results.
3. **Recall at the operating threshold**, because a missed cardiomegaly is the
   costly error.
4. **Accuracy**, reported last and always next to the majority-class baseline so
   it cannot be read as evidence of anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..common.config import Config
from ..common.io_utils import save_dataframe, save_json
from ..common.logging_utils import get_logger
from ..common.metrics import binary_metrics, bootstrap_ci, find_best_threshold
from ..common.paths import ensure_dir
from ..common.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve

__all__ = [
    "predict_probabilities",
    "tune_threshold",
    "evaluate_binary",
    "export_errors",
    "build_comparison_table",
    "select_error_examples",
    "recommend_model",
    "plot_evaluation_figures",
]

logger = get_logger("xray.evaluate")


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader.

    Returns:
        ``(probabilities, targets, indices)``. ``indices`` are row positions into
        the split frame, so a prediction can be traced back to its file.
    """
    from ..common.compat import autocast_context

    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []

    for batch in loader:
        images, batch_targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        with autocast_context(use_amp, device.type):
            logits = model(images)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        targets.append(batch_targets.numpy().ravel())
        indices.append(batch[2].numpy().ravel() if len(batch) > 2
                       else np.arange(len(batch_targets)))

    return (np.concatenate(probabilities).astype(np.float32),
            np.concatenate(targets).astype(int),
            np.concatenate(indices).astype(int))


def tune_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    cfg: Config,
) -> tuple[float, dict[str, Any]]:
    """Choose the operating threshold on the VALIDATION split.

    Strategy from ``evaluation.threshold_strategy``:

    * ``max_f1`` — balances precision and recall. Sensible default here, because
      at low prevalence a threshold chosen for accuracy collapses to "predict
      negative always".
    * ``youden`` — maximises sensitivity + specificity - 1. Weights a missed case
      and a false alarm equally, ignoring how many more negatives there are.
    * ``fixed`` — use ``evaluation.fixed_threshold``.

    The default 0.5 is almost always wrong after ``pos_weight`` training: the loss
    weighting deliberately shifts the logit scale, so the natural operating point
    moves with it.
    """
    strategy = str(cfg.evaluation.get("threshold_strategy", "max_f1")).lower()
    fixed = float(cfg.evaluation.get("fixed_threshold", 0.5))

    if strategy == "fixed":
        return fixed, {"strategy": "fixed", "threshold": fixed, "tuned_on": "none"}

    metric = "youden" if "youden" in strategy else "f1"
    threshold, score = find_best_threshold(y_val, p_val, metric=metric)

    info = {
        "strategy": strategy,
        "objective": metric,
        "threshold": round(float(threshold), 4),
        "objective_value": round(float(score), 4),
        "tuned_on": "validation",
        "n_validation": int(len(y_val)),
        "validation_prevalence": round(float(np.mean(y_val)), 5),
        "note": "Tuned on validation only. Test metrics are reported at this threshold "
                "AND at 0.5, so the effect of tuning is visible.",
    }
    logger.info("Threshold tuned on validation (%s): %.3f (%s = %.4f)",
                strategy, threshold, metric, score)
    return float(threshold), info


def evaluate_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cfg: Config,
    split_name: str = "test",
    with_ci: bool = True,
) -> dict[str, Any]:
    """Compute the full metric block for a binary split, with bootstrap CIs."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    tuned = binary_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)
    tuned["threshold"] = float(threshold)
    default = binary_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
    default["threshold"] = 0.5

    prevalence = float(y_true.mean())
    block: dict[str, Any] = {
        "split": split_name,
        "n_images": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": round(prevalence, 5),
        "at_tuned_threshold": tuned,
        "at_default_threshold_0.5": default,
        # Chance level for PR-AUC is the prevalence, not 0.5. Stored explicitly so
        # nobody has to remember it when reading the numbers.
        "pr_auc_chance_level": round(prevalence, 5),
        "pr_auc_lift_over_chance": round(
            float(tuned.get("pr_auc", float("nan")) / max(prevalence, 1e-9)), 3),
        "accuracy_of_always_negative": round(1 - prevalence, 5),
    }

    if with_ci and bool(cfg.evaluation.get("bootstrap_ci", True)):
        n_boot = int(cfg.evaluation.get("n_bootstrap", 1000))
        block["confidence_intervals_95pct"] = {
            "roc_auc": bootstrap_ci(y_true, y_prob, roc_auc_score,
                                    n_bootstrap=n_boot, seed=int(cfg.seed)),
            "pr_auc": bootstrap_ci(y_true, y_prob, average_precision_score,
                                   n_bootstrap=n_boot, seed=int(cfg.seed)),
        }

    logger.info(
        "[%s] PR-AUC %.4f (chance %.4f, lift %.1fx) | ROC-AUC %.4f | recall %.3f | "
        "precision %.3f | accuracy %.4f",
        split_name, tuned.get("pr_auc", float("nan")), prevalence,
        block["pr_auc_lift_over_chance"], tuned.get("roc_auc", float("nan")),
        tuned["recall"], tuned["precision"], tuned["accuracy"],
    )
    logger.info(
        "    For scale: predicting 'no cardiomegaly' for every image would score "
        "%.4f accuracy and find 0 of %d cases.",
        block["accuracy_of_always_negative"], int(y_true.sum()),
    )
    return block


def select_error_examples(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_per_category: int = 3,
) -> dict[str, list[int]]:
    """Pick representative TP / TN / FP / FN cases for Grad-CAM inspection.

    Within each category the *most confident* examples are chosen. A false
    negative at p = 0.02 is far more informative than one at p = 0.49: the second
    is a borderline call, the first is the model being certain and wrong.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    categories = {
        "TP": np.flatnonzero((y_true == 1) & (y_pred == 1)),
        "TN": np.flatnonzero((y_true == 0) & (y_pred == 0)),
        "FP": np.flatnonzero((y_true == 0) & (y_pred == 1)),
        "FN": np.flatnonzero((y_true == 1) & (y_pred == 0)),
    }

    selection: dict[str, list[int]] = {}
    for name, positions in categories.items():
        if positions.size == 0:
            continue
        # Confidently wrong / confidently right first.
        descending = name in {"TP", "FP"}
        order = np.argsort(y_prob[positions])
        if descending:
            order = order[::-1]
        selection[name] = positions[order][:n_per_category].tolist()
    return selection


def export_errors(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cfg: Config,
    out_dir: Path | str,
    prefix: str = "test",
    n_examples: int = 10,
) -> dict[str, Any]:
    """Export per-image predictions and the misclassified cases.

    The co-occurring findings on false positives are worth looking at: if the
    model consistently flags images that also carry Effusion or Cardiomegaly-
    adjacent findings, it may be keying on something correlated with cardiomegaly
    rather than heart size itself.
    """
    out = ensure_dir(out_dir)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    image_column = str(cfg.dataset.image_column)
    label_column = str(cfg.dataset.label_column)
    patient_column = str(cfg.dataset.patient_column)

    predictions = pd.DataFrame({
        "image": frame[image_column].to_numpy(),
        "patient_id": frame[patient_column].to_numpy(),
        "y_true": y_true,
        "y_pred": y_pred,
        "probability": np.round(y_prob, 4),
        "margin": np.round(np.abs(y_prob - threshold), 4),
    })
    if label_column in frame.columns:
        predictions["findings"] = frame[label_column].to_numpy()

    predictions["error_type"] = np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 0),
         (y_true == 0) & (y_pred == 1), (y_true == 1) & (y_pred == 0)],
        ["TP", "TN", "FP", "FN"], default="?",
    )
    predictions["correct"] = predictions.y_true == predictions.y_pred

    save_dataframe(predictions, out / f"{prefix}_predictions.csv")
    errors = predictions[~predictions.correct].sort_values("margin", ascending=False)
    save_dataframe(errors, out / f"{prefix}_errors.csv")

    false_negatives = errors[errors.error_type == "FN"]
    false_positives = errors[errors.error_type == "FP"]

    summary: dict[str, Any] = {
        "split": prefix,
        "threshold": float(threshold),
        "n_images": int(len(predictions)),
        "n_errors": int(len(errors)),
        "error_rate": round(float(len(errors) / max(len(predictions), 1)), 4),
        "counts": {k: int(v) for k, v in predictions.error_type.value_counts().items()},
        "most_confident_false_negatives": false_negatives.nsmallest(
            n_examples, "probability")[["image", "probability"]].to_dict(orient="records"),
        "most_confident_false_positives": false_positives.nlargest(
            n_examples, "probability")[["image", "probability"]].to_dict(orient="records"),
        "files": {"predictions": f"{prefix}_predictions.csv", "errors": f"{prefix}_errors.csv"},
    }

    if "findings" in predictions.columns and len(false_positives):
        co_occurring: dict[str, int] = {}
        for entry in false_positives["findings"].fillna(""):
            for finding in str(entry).split("|"):
                finding = finding.strip()
                if finding:
                    co_occurring[finding] = co_occurring.get(finding, 0) + 1
        summary["false_positive_findings"] = dict(
            sorted(co_occurring.items(), key=lambda kv: -kv[1])[:10]
        )
        summary["false_positive_findings_note"] = (
            "Findings that appear on images the model wrongly flagged. A finding that "
            "dominates this list may be what the model is actually keying on."
        )

    logger.info("[%s] %d errors of %d (%.1f%%): %d FN, %d FP",
                prefix, len(errors), len(predictions), 100 * summary["error_rate"],
                len(false_negatives), len(false_positives))

    save_json(summary, out / f"{prefix}_error_summary.json")
    return summary


def plot_evaluation_figures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_dir: Path | str,
    model_name: str = "DenseNet121",
    extra_curves: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Path]:
    """Confusion matrix, ROC and PR curves for the test split."""
    out = ensure_dir(out_dir)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    curves = {model_name: (y_true, y_prob)}
    if extra_curves:
        curves.update(extra_curves)

    return {
        "confusion": plot_confusion_matrix(
            y_true, y_pred, out / "confusion_matrix.png",
            class_names=["No cardiomegaly", "Cardiomegaly"],
            title=f"{model_name} — test (threshold = {threshold:.2f})"),
        "confusion_normalized": plot_confusion_matrix(
            y_true, y_pred, out / "confusion_matrix_normalized.png",
            class_names=["No cardiomegaly", "Cardiomegaly"], normalize="true",
            title=f"{model_name} — row-normalised (recall view)"),
        "roc": plot_roc_curve(curves, out / "roc_curve.png", title="ROC — test split"),
        "pr": plot_pr_curve(curves, out / "pr_curve.png",
                            title="Precision-Recall — test split"),
    }


def recommend_model(
    results: Mapping[str, Mapping[str, Any]],
    metric: str = "pr_auc",
    min_meaningful_gain: float = 0.02,
    complexity_order: Sequence[str] = ("majority_class", "logreg_pixel_features",
                                       "densenet121"),
) -> dict[str, Any]:
    """Decide whether the deep model earned its complexity.

    The brief asks the baselines to establish that DenseNet121 "actually provides
    useful predictive capability". This makes that check explicit rather than
    leaving it to a reader comparing rows by eye.

    A more complex model is recommended only when it beats every simpler one by
    more than ``min_meaningful_gain`` PR-AUC. The bar is 0.02 rather than the ECG
    pipeline's 0.01 because PR-AUC on a small, imbalanced test split is noisier
    than macro AUC — with a few dozen positives, a couple of rank swaps move it
    by more than a point.

    Args:
        results: ``{model_name: {"metrics": {...}, "parameters": int}}``.
        metric: Ranking metric, read from the tuned-threshold block.
        min_meaningful_gain: Minimum improvement that counts.
        complexity_order: Models from simplest to most complex.

    Returns:
        A decision record with the recommendation and its justification.
    """
    def score_of(block: Mapping[str, Any]) -> float:
        metrics = block.get("metrics", {})
        tuned = metrics.get("at_tuned_threshold", metrics)
        return float(tuned.get(metric, float("nan")))

    scores = {name: score_of(block) for name, block in results.items()}
    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    if not valid:
        return {"recommended": None, "reason": f"No model reported {metric}."}

    def complexity(name: str) -> int:
        return complexity_order.index(name) if name in complexity_order else len(complexity_order)

    ordered = sorted(valid, key=complexity)
    recommended = ordered[0]
    reasons: list[str] = []

    for candidate in ordered[1:]:
        gain = valid[candidate] - valid[recommended]
        if gain > min_meaningful_gain:
            reasons.append(
                f"{candidate} beats {recommended} by {gain:+.4f} {metric}, above the "
                f"{min_meaningful_gain} bar — the added complexity is justified."
            )
            recommended = candidate
        else:
            reasons.append(
                f"{candidate} gains only {gain:+.4f} {metric} over {recommended}, within "
                f"the {min_meaningful_gain} noise bar — not enough to justify the extra "
                f"parameters and training cost."
            )

    decision = {
        "metric": metric,
        "min_meaningful_gain": min_meaningful_gain,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "parameters": {k: block.get("parameters") for k, block in results.items()},
        "recommended": recommended,
        "reasoning": reasons,
    }

    logger.info("RECOMMENDATION: keep %s (%s = %.4f)", recommended, metric, valid[recommended])
    for reason in reasons:
        logger.info("    %s", reason)

    if recommended == "logreg_pixel_features":
        decision["warning"] = (
            "A logistic regression on a 32x32 thumbnail and an intensity histogram matched "
            "or beat DenseNet121. That means the task is being solved largely by global "
            "exposure and body size rather than cardiac silhouette shape. Do not report "
            "the CNN as an improvement until this is resolved — check the training curves "
            "for underfitting and confirm the Grad-CAM maps actually sit on the heart."
        )
        logger.warning(decision["warning"])
    elif recommended == "majority_class":
        decision["warning"] = (
            "No model beat the majority-class predictor on PR-AUC. Nothing here has "
            "learned to detect the target."
        )
        logger.warning(decision["warning"])
    return decision


def build_comparison_table(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Comparison across X-ray experiments.

    Accuracy is included but placed **last**, deliberately: it is only meaningful
    when read next to the majority-class row, which will show a high accuracy with
    zero recall.
    """
    rows = []
    for name, block in results.items():
        metrics = block.get("metrics", {})
        tuned = metrics.get("at_tuned_threshold", metrics)
        rows.append({
            "model": name,
            "pr_auc": round(float(tuned.get("pr_auc", float("nan"))), 4),
            "roc_auc": round(float(tuned.get("roc_auc", float("nan"))), 4),
            "recall": round(float(tuned.get("recall", float("nan"))), 4),
            "precision": round(float(tuned.get("precision", float("nan"))), 4),
            "f1": round(float(tuned.get("f1", float("nan"))), 4),
            "specificity": round(float(tuned.get("specificity", float("nan"))), 4),
            "accuracy": round(float(tuned.get("accuracy", float("nan"))), 4),
            "parameters": block.get("parameters"),
            "train_seconds": block.get("train_seconds"),
        })
    return pd.DataFrame(rows)
