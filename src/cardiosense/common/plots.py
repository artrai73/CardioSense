"""Figure helpers shared by all three pipelines.

Every function saves to disk and returns the ``Path``, so a training script never
leaves a figure open and Colab never silently drops it on disconnect. Matplotlib
runs headless via the ``Agg`` backend unless a display is already configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

if not matplotlib.get_backend().lower().startswith(("module://", "qt", "tk", "macosx", "nbagg")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

__all__ = [
    "save_figure",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_training_curves",
    "plot_class_distribution",
]

DEFAULT_DPI = 150
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = DEFAULT_DPI, close: bool = True) -> Path:
    """Save *fig* to *path*, creating parents, and close it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return target


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path | str,
    class_names: Sequence[str] | None = None,
    normalize: str | None = None,
    title: str = "Confusion matrix",
) -> Path:
    """Plot and save a confusion matrix.

    Args:
        normalize: ``None``, ``"true"`` (row-wise recall view), ``"pred"`` or ``"all"``.
            For imbalanced data, ``"true"`` is far more readable than raw counts.
    """
    labels = list(range(len(class_names))) if class_names is not None else None
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=True,
                 values_format=".2f" if normalize else "d")
    ax.set_title(title)
    ax.grid(False)
    return save_figure(fig, path)


def plot_roc_curve(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path | str,
    title: str = "ROC curve",
) -> Path:
    """Plot one or more ROC curves.

    Args:
        curves: ``{label: (y_true, y_score)}``. Multiple entries overlay models
            (e.g. Logistic Regression vs XGBoost) on one axis.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for label, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return save_figure(fig, path)


def plot_pr_curve(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path | str,
    title: str = "Precision-Recall curve",
) -> Path:
    """Plot precision-recall curves with the prevalence baseline marked.

    On an imbalanced target the chance level for PR-AUC is the positive
    prevalence, not 0.5 — drawing it prevents over-reading a modest PR-AUC.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    prevalence = None
    for label, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, lw=2, label=f"{label} (AP = {auc(recall, precision):.3f})")
        prevalence = float(np.mean(np.asarray(y_true)))
    if prevalence is not None:
        ax.axhline(prevalence, color="k", ls="--", lw=1,
                   label=f"Chance (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return save_figure(fig, path)


def plot_calibration_curve(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path | str,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Calibration (reliability diagram)",
) -> Path:
    """Reliability diagram plus a histogram of predicted probabilities.

    The lower panel matters: a curve drawn from a bin holding three samples looks
    identical to one holding three hundred, and only the histogram reveals which.
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_top.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for label, (y_true, y_prob) in curves.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )
        ax_top.plot(mean_pred, frac_pos, "o-", lw=2, ms=5, label=label)
        ax_bottom.hist(y_prob, range=(0, 1), bins=n_bins, histtype="step", lw=2, label=label)

    ax_top.set_ylabel("Observed frequency")
    ax_top.set_title(title)
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.set_ylim(-0.02, 1.02)

    ax_bottom.set_xlabel("Predicted probability")
    ax_bottom.set_ylabel("Count")
    ax_bottom.set_xlim(0, 1)
    return save_figure(fig, path)


def plot_training_curves(
    history: Mapping[str, Sequence[float]],
    path: Path | str,
    loss_keys: Sequence[str] = ("train_loss", "val_loss"),
    metric_keys: Sequence[str] = (),
    best_epoch: int | None = None,
    title: str = "Training history",
) -> Path:
    """Plot loss (left) and monitored metrics (right) against epoch.

    Args:
        history: ``{key: [value_per_epoch]}`` as produced by the training loops.
        best_epoch: 1-based epoch of the restored checkpoint; drawn as a marker.
    """
    metric_keys = [k for k in metric_keys if k in history]
    n_panels = 2 if metric_keys else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False)

    ax_loss = axes[0][0]
    for key in loss_keys:
        if key in history:
            values = list(history[key])
            ax_loss.plot(range(1, len(values) + 1), values, lw=2, label=key)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(fontsize=9)

    if metric_keys:
        ax_metric = axes[0][1]
        for key in metric_keys:
            values = list(history[key])
            ax_metric.plot(range(1, len(values) + 1), values, lw=2, label=key)
        ax_metric.set_xlabel("Epoch")
        ax_metric.set_ylabel("Metric")
        ax_metric.set_title("Monitored metrics")
        ax_metric.legend(fontsize=9)

    if best_epoch is not None:
        for row in axes:
            for ax in row:
                ax.axvline(best_epoch, color="crimson", ls=":", lw=1.5,
                           label=f"best epoch = {best_epoch}")
                ax.legend(fontsize=9)

    fig.suptitle(title)
    return save_figure(fig, path)


def plot_class_distribution(
    counts: Mapping[str, int],
    path: Path | str,
    title: str = "Class distribution",
    horizontal: bool = False,
) -> Path:
    """Bar chart of label counts with the percentage annotated on each bar."""
    labels = list(counts)
    values = [counts[k] for k in labels]
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4.5))
    if horizontal:
        bars = ax.barh(labels, values, color="steelblue")
        ax.set_xlabel("Count")
    else:
        bars = ax.bar(labels, values, color="steelblue")
        ax.set_ylabel("Count")

    for bar, value in zip(bars, values):
        pct = 100 * value / total
        if horizontal:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {value} ({pct:.1f}%)", va="center", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.margins(y=0.15 if not horizontal else 0.05, x=0.05 if not horizontal else 0.15)
    return save_figure(fig, path)


def new_figure(*args: Any, **kwargs: Any) -> tuple[plt.Figure, Any]:
    """Thin wrapper so pipeline modules do not import pyplot directly."""
    return plt.subplots(*args, **kwargs)
