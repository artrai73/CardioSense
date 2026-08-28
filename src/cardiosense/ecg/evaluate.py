"""Evaluation for the multi-label ECG task.

The single most important rule here: **plain accuracy is never reported.** With
multi-label targets, "accuracy" is ambiguous — it could mean per-element accuracy
(which is inflated by the many easy negatives) or exact-match ratio (which is
brutally strict). Both are misleading if labelled simply "accuracy", so this
module reports exact-match under that explicit name and leads with macro ROC-AUC.

Thresholds are tuned **per class on validation**. A single global 0.5 is wrong
here: the classes have prevalences from ~12% (HYP) to ~44% (NORM), and after
``pos_weight`` training the logit scales differ between them.
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
from ..common.metrics import bootstrap_ci, find_best_threshold, multilabel_metrics
from ..common.paths import ensure_dir
from ..common.plots import save_figure

__all__ = [
    "predict_probabilities",
    "tune_per_class_thresholds",
    "evaluate_multilabel",
    "plot_per_class_curves",
    "plot_per_class_confusion",
    "export_errors",
    "build_comparison_table",
    "recommend_model",
    "per_class_table",
]

logger = get_logger("ecg.evaluate")


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return ``(probabilities, labels)``.

    The model emits logits; the sigmoid is applied here. AMP is optional and only
    affects speed — the outputs are cast back to float32 either way.
    """
    from ..common.compat import autocast_context

    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for waveforms, batch_labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        with autocast_context(use_amp, device.type):
            logits = model(waveforms)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(batch_labels.numpy())

    return (np.vstack(probabilities).astype(np.float32),
            np.vstack(labels).astype(np.float32))


def tune_per_class_thresholds(
    y_val: np.ndarray,
    p_val: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick one decision threshold per class, on the VALIDATION split.

    Each class is tuned independently because each is an independent binary
    decision in a multi-label setting. Strategy comes from
    ``evaluation.threshold_strategy``:

    * ``per_class_f1`` — maximise F1 for that class
    * ``per_class_youden`` — maximise sensitivity + specificity - 1
    * ``fixed`` — use ``evaluation.default_threshold`` for every class

    Returns:
        ``(thresholds, info)`` with one threshold per class in class order.
    """
    classes = list(cfg.task.classes)
    strategy = str(cfg.evaluation.get("threshold_strategy", "per_class_f1")).lower()
    default = float(cfg.evaluation.get("default_threshold", 0.5))

    if strategy == "fixed":
        thresholds = np.full(len(classes), default, dtype=float)
        return thresholds, {"strategy": "fixed", "tuned_on": "none",
                            "thresholds": dict(zip(classes, thresholds.tolist()))}

    metric = "youden" if "youden" in strategy else "f1"
    thresholds = np.zeros(len(classes), dtype=float)
    details: dict[str, Any] = {}

    for index, name in enumerate(classes):
        column_true = y_val[:, index]
        if len(np.unique(column_true)) < 2:
            thresholds[index] = default
            details[name] = {"threshold": default, "note": "class absent from validation"}
            continue
        threshold, score = find_best_threshold(column_true, p_val[:, index], metric=metric)
        thresholds[index] = threshold
        details[name] = {"threshold": round(float(threshold), 4),
                         "objective": metric,
                         "objective_value": round(float(score), 4)}

    logger.info("Per-class thresholds (%s, tuned on validation): %s",
                metric, {c: round(float(t), 3) for c, t in zip(classes, thresholds)})
    return thresholds, {"strategy": strategy, "objective": metric,
                        "tuned_on": "validation", "per_class": details,
                        "thresholds": dict(zip(classes, thresholds.round(4).tolist()))}


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | float,
    cfg: Config,
    split_name: str = "test",
    with_ci: bool = True,
) -> dict[str, Any]:
    """Compute the full multi-label metric block, with per-class bootstrap CIs.

    Reported: macro/micro ROC-AUC, macro PR-AUC, macro/micro/weighted F1, Hamming
    loss, exact-match ratio, and a per-class breakdown. Not reported: anything
    called "accuracy".
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    classes = list(cfg.task.classes)
    metrics = multilabel_metrics(y_true, y_prob, thresholds=thresholds, class_names=classes)
    metrics["split"] = split_name

    if with_ci and bool(cfg.evaluation.get("bootstrap_ci", True)):
        n_boot = int(cfg.evaluation.get("n_bootstrap", 1000))
        for index, name in enumerate(classes):
            if len(np.unique(y_true[:, index])) < 2:
                continue
            metrics["per_class"][name]["roc_auc_ci"] = bootstrap_ci(
                y_true[:, index], y_prob[:, index], roc_auc_score,
                n_bootstrap=n_boot, seed=int(cfg.seed),
            )
            metrics["per_class"][name]["pr_auc_ci"] = bootstrap_ci(
                y_true[:, index], y_prob[:, index], average_precision_score,
                n_bootstrap=n_boot, seed=int(cfg.seed),
            )

    logger.info("[%s] macro ROC-AUC %.4f | macro PR-AUC %.4f | macro F1 %.4f | exact-match %.4f",
                split_name, metrics["macro_roc_auc"], metrics["macro_pr_auc"],
                metrics["macro_f1"], metrics["exact_match"])
    for name in classes:
        entry = metrics["per_class"][name]
        logger.info("    %-5s AUC %.3f | PR-AUC %.3f | F1 %.3f | support %d",
                    name, entry.get("roc_auc", float("nan")),
                    entry.get("pr_auc", float("nan")), entry["f1"], entry["support"])
    return metrics


def per_class_table(metrics: Mapping[str, Any], classes: Sequence[str]) -> pd.DataFrame:
    """Flatten the per-class block into a tidy table for the report."""
    rows = []
    for name in classes:
        entry = metrics["per_class"][name]
        row = {
            "class": name,
            "support": entry["support"],
            "prevalence": round(entry["prevalence"], 4),
            "threshold": entry["threshold"],
            "roc_auc": round(entry.get("roc_auc", float("nan")), 4),
            "pr_auc": round(entry.get("pr_auc", float("nan")), 4),
            "precision": round(entry["precision"], 4),
            "recall": round(entry["recall"], 4),
            "f1": round(entry["f1"], 4),
        }
        ci = entry.get("roc_auc_ci")
        if ci:
            row["roc_auc_95ci"] = f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_per_class_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: Sequence[str],
    out_dir: Path | str,
    prefix: str = "test",
) -> dict[str, Path]:
    """One figure with per-class ROC curves, one with per-class PR curves.

    The PR figure marks each class's prevalence, because PR-AUC's chance level is
    the positive rate — without that line a PR-AUC of 0.30 for a 12%-prevalence
    class looks like failure when it is nearly triple chance.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    out = ensure_dir(out_dir)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for index, name in enumerate(classes):
        if len(np.unique(y_true[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, index], y_prob[:, index])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC per diagnostic superclass — {prefix}")
    ax.legend(loc="lower right", fontsize=9)
    paths["roc"] = save_figure(fig, out / "roc_curve.png")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for index, name in enumerate(classes):
        if len(np.unique(y_true[:, index])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, index], y_prob[:, index])
        prevalence = float(y_true[:, index].mean())
        line, = ax.plot(recall, precision, lw=2,
                        label=f"{name} (AP = {auc(recall, precision):.3f}, "
                              f"prev = {prevalence:.2f})")
        ax.axhline(prevalence, color=line.get_color(), ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall per superclass — {prefix}\n"
                 "dotted lines = per-class chance level (prevalence)")
    ax.legend(loc="best", fontsize=8)
    paths["pr"] = save_figure(fig, out / "pr_curve.png")
    return paths


def plot_per_class_confusion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    classes: Sequence[str],
    out_dir: Path | str,
    filename: str = "confusion_matrix.png",
) -> Path:
    """One 2x2 confusion matrix per class, in a single figure.

    A multi-label problem has no single N-by-N confusion matrix: a record can
    belong to several classes at once, so the rows would not sum to one. Five
    independent binary matrices is the honest presentation.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    out = ensure_dir(out_dir)
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    fig, axes = plt.subplots(1, len(classes), figsize=(3.1 * len(classes), 3.4))
    axes = np.atleast_1d(axes).ravel()
    for index, (ax, name) in enumerate(zip(axes, classes)):
        matrix = confusion_matrix(y_true[:, index], y_pred[:, index], labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                        fontsize=10)
        ax.set_xticks([0, 1], ["pred -", "pred +"])
        ax.set_yticks([0, 1], ["true -", "true +"])
        ax.set_title(f"{name}\nthreshold {thresholds[index]:.2f}", fontsize=10)
        ax.grid(False)
    fig.suptitle("Per-class confusion matrices (multi-label: one binary decision per class)")
    return save_figure(fig, out / filename)


def export_errors(
    database: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    classes: Sequence[str],
    out_dir: Path | str,
    prefix: str = "test",
    n_examples: int = 10,
) -> dict[str, Any]:
    """Export misclassified records for inspection.

    Writes a per-record CSV with each class's probability, prediction and error
    type, plus a summary listing the most confident errors per class. The
    confident false negatives are the interesting ones: an MI the model scored at
    0.05 is a failure worth looking at the waveform for.

    Args:
        database: Split metadata, row-aligned with ``y_true``.
        y_true: Multi-hot labels.
        y_prob: Predicted probabilities.
        thresholds: Per-class thresholds.
        classes: Class names in column order.
        out_dir: Output directory.
        prefix: Filename prefix.
        n_examples: Confident errors to list per class.
    """
    out = ensure_dir(out_dir)
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    frame = pd.DataFrame({"ecg_id": database.index.to_numpy()})
    for column in ("patient_id", "age", "sex", "strat_fold"):
        if column in database.columns:
            frame[column] = database[column].to_numpy()

    for index, name in enumerate(classes):
        frame[f"{name}_true"] = y_true[:, index].astype(int)
        frame[f"{name}_prob"] = np.round(y_prob[:, index], 4)
        frame[f"{name}_pred"] = y_pred[:, index]
        frame[f"{name}_error"] = np.select(
            [(y_true[:, index] == 1) & (y_pred[:, index] == 0),
             (y_true[:, index] == 0) & (y_pred[:, index] == 1)],
            ["FN", "FP"], default="",
        )

    frame["n_errors"] = sum(
        (frame[f"{name}_error"] != "").astype(int) for name in classes
    )
    frame["exact_match"] = (frame["n_errors"] == 0).astype(int)
    save_dataframe(frame, out / f"{prefix}_predictions.csv")

    errors = frame[frame.n_errors > 0].sort_values("n_errors", ascending=False)
    save_dataframe(errors, out / f"{prefix}_errors.csv")

    summary: dict[str, Any] = {
        "split": prefix,
        "n_records": int(len(frame)),
        "n_records_with_any_error": int(len(errors)),
        "exact_match_rate": round(float(frame.exact_match.mean()), 4),
        "per_class": {},
        "files": {"predictions": f"{prefix}_predictions.csv", "errors": f"{prefix}_errors.csv"},
    }

    for index, name in enumerate(classes):
        error_column = frame[f"{name}_error"]
        false_negatives = frame[error_column == "FN"].nsmallest(n_examples, f"{name}_prob")
        false_positives = frame[error_column == "FP"].nlargest(n_examples, f"{name}_prob")
        summary["per_class"][name] = {
            "n_false_negative": int((error_column == "FN").sum()),
            "n_false_positive": int((error_column == "FP").sum()),
            "threshold": round(float(thresholds[index]), 4),
            "most_confident_false_negatives": false_negatives[
                ["ecg_id", f"{name}_prob"]].to_dict(orient="records"),
            "most_confident_false_positives": false_positives[
                ["ecg_id", f"{name}_prob"]].to_dict(orient="records"),
        }

    logger.info("[%s] %d of %d records have at least one error (exact-match %.3f)",
                prefix, len(errors), len(frame), summary["exact_match_rate"])
    save_json(summary, out / f"{prefix}_error_summary.json")
    return summary


def recommend_model(
    results: Mapping[str, Mapping[str, Any]],
    metric: str = "macro_roc_auc",
    min_meaningful_gain: float = 0.01,
    complexity_order: Sequence[str] = ("statistical_baseline", "cnn1d", "resnet1d"),
) -> dict[str, Any]:
    """Recommend which model to keep, defending the simpler one by default.

    Ranks by *metric*, then asks whether the extra complexity earned its place. A
    more complex model is recommended only when it beats every simpler model by
    more than ``min_meaningful_gain``.

    Why 0.01 macro AUC as the bar: run-to-run variation from seeding alone is
    typically a few thousandths of an AUC point on this task, so a gain smaller
    than 0.01 is not distinguishable from noise without repeating the run across
    seeds. Reporting a deeper model as "better" on a 0.003 gain is exactly the
    kind of claim that fails to replicate.

    This is the check the Phase 1 brief asks for: if ResNet-1D provides no
    meaningful benefit over the plain CNN, say so and keep the CNN.

    Args:
        results: ``{model_name: {"metrics": {...}, "parameters": int}}``.
        metric: Ranking metric.
        min_meaningful_gain: Minimum improvement that counts.
        complexity_order: Models from simplest to most complex.

    Returns:
        A decision record with the recommendation and its justification.
    """
    scores = {
        name: float(block.get("metrics", {}).get(metric, float("nan")))
        for name, block in results.items()
    }
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

    if recommended == "statistical_baseline":
        decision["warning"] = (
            "The classical baseline matched or beat the deep models. Either the network is "
            "undertrained, or the task is being solved by global signal statistics. Do not "
            "report the CNN as an improvement until this is resolved — check the training "
            "curves for underfitting before increasing model capacity."
        )
        logger.warning(decision["warning"])
    return decision


def build_comparison_table(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Comparison table across ECG experiments (baseline / CNN / ResNet)."""
    rows = []
    for name, block in results.items():
        metrics = block.get("metrics", {})
        rows.append({
            "model": name,
            "macro_roc_auc": round(float(metrics.get("macro_roc_auc", float("nan"))), 4),
            "macro_pr_auc": round(float(metrics.get("macro_pr_auc", float("nan"))), 4),
            "macro_f1": round(float(metrics.get("macro_f1", float("nan"))), 4),
            "micro_f1": round(float(metrics.get("micro_f1", float("nan"))), 4),
            "exact_match": round(float(metrics.get("exact_match", float("nan"))), 4),
            "hamming_loss": round(float(metrics.get("hamming_loss", float("nan"))), 4),
            "parameters": block.get("parameters"),
            "train_seconds": block.get("train_seconds"),
        })
    return pd.DataFrame(rows)
