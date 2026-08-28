"""SHAP explanations for the clinical model.

Global and local explanations, with the explainer chosen to match the model:

* **XGBoost** -> ``TreeExplainer``. Exact (not sampled) and fast for tree
  ensembles.
* **Logistic Regression** -> ``LinearExplainer``, with the training set as the
  background distribution.
* Anything else -> the generic ``shap.Explainer``, which falls back to a
  model-agnostic method.

What SHAP does and does not tell you
------------------------------------

A SHAP value is the contribution of a feature to the difference between this
prediction and the average prediction, averaged fairly over all orderings in
which features could be added. It describes **the model**, faithfully.

It does not describe the disease. If two features are correlated — and here
``thalach`` and ``age`` are — the model may load its reliance onto one of them
arbitrarily, and SHAP will report that arbitrary split honestly. A feature with
near-zero SHAP importance is therefore not proven clinically irrelevant; it may
simply be redundant given another feature the model preferred. This caveat is
written into the saved outputs so it travels with the figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..common.config import Config
from ..common.io_utils import save_dataframe, save_json
from ..common.logging_utils import get_logger
from ..common.paths import ensure_dir
from ..common.plots import save_figure

__all__ = ["build_explainer", "explain_global", "explain_local", "run_shap_analysis"]

logger = get_logger(__name__)

_CAVEAT = (
    "SHAP values explain the MODEL, not the disease. They show how each feature moved this "
    "model's output relative to its average output. Correlated features share credit "
    "arbitrarily, so a low SHAP importance does not establish clinical irrelevance, and a "
    "high one does not establish causation."
)


def build_explainer(model: Any, X_background: np.ndarray, feature_names: Sequence[str]) -> Any:
    """Build the SHAP explainer appropriate to the model type.

    Args:
        model: The fitted estimator.
        X_background: Background/reference data (use the training matrix).
        feature_names: Names for the transformed columns.

    Returns:
        A SHAP explainer.

    Raises:
        ImportError: If shap is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for explanations. Run: pip install shap") from exc

    model_name = type(model).__name__

    if model_name.startswith("XGB"):
        logger.info("Using shap.TreeExplainer (exact for tree ensembles).")
        return shap.TreeExplainer(model, feature_names=list(feature_names))

    if model_name == "LogisticRegression":
        logger.info("Using shap.LinearExplainer with the training set as background.")
        return shap.LinearExplainer(model, X_background, feature_names=list(feature_names))

    logger.info("Model %s has no specialised explainer; falling back to shap.Explainer.",
                model_name)
    return shap.Explainer(model, X_background, feature_names=list(feature_names))


def _to_explanation(explainer: Any, X: np.ndarray, feature_names: Sequence[str]) -> Any:
    """Compute SHAP values and normalise them to a 2-D ``shap.Explanation``.

    Different explainers and model types return different shapes — a plain array,
    a list of one array per class, or a 3-D array with a trailing class axis. For
    a binary classifier we always want the positive class, shape
    ``(n_samples, n_features)``.
    """
    import shap

    explanation = explainer(X)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    if values.ndim == 3:
        # (n, features, classes) -> positive class
        values = values[:, :, -1]
        base = base[:, -1] if base.ndim == 2 else base
    elif isinstance(explanation.values, list):  # pragma: no cover - legacy shap
        values = np.asarray(explanation.values[-1])
        base = np.asarray(explanation.base_values[-1])

    if base.ndim == 0:
        base = np.repeat(float(base), values.shape[0])

    return shap.Explanation(
        values=values,
        base_values=base,
        data=np.asarray(X),
        feature_names=list(feature_names),
    )


def explain_global(
    explanation: Any,
    out_dir: Path | str,
    max_display: int = 15,
) -> pd.DataFrame:
    """Produce global explanations: beeswarm summary and mean-|SHAP| bar chart.

    Args:
        explanation: A 2-D ``shap.Explanation``.
        out_dir: Output directory.
        max_display: Features to show.

    Returns:
        A DataFrame of features ranked by mean absolute SHAP value.
    """
    import matplotlib.pyplot as plt
    import shap

    out = ensure_dir(out_dir)

    # Beeswarm: shows magnitude, direction and the feature value together, which
    # is the single most informative global SHAP plot.
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary — impact on predicted risk", y=1.02, fontsize=11)
    save_figure(fig, out / "shap_summary.png")

    # Bar chart of mean |SHAP|: the cleanest importance ranking for a report table.
    plt.figure()
    shap.plots.bar(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle("Mean |SHAP| — global feature importance", y=1.02, fontsize=11)
    save_figure(fig, out / "shap_importance_bar.png")

    values = np.asarray(explanation.values)
    ranking = pd.DataFrame({
        "feature": list(explanation.feature_names),
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking.round(5)

    save_dataframe(ranking, out / "shap_feature_importance.csv")
    logger.info("Top SHAP features: %s", ", ".join(ranking.feature.head(5)))
    return ranking


def explain_local(
    explanation: Any,
    indices: Sequence[int],
    labels: Sequence[str],
    out_dir: Path | str,
    max_display: int = 12,
) -> list[dict[str, Any]]:
    """Produce per-patient waterfall plots.

    Args:
        explanation: A 2-D ``shap.Explanation`` over the split being explained.
        indices: Row positions to explain.
        labels: A short descriptor per index, e.g. ``"TP"``, ``"FN"``; used in
            the filename and title so the figures are self-describing.
        out_dir: Output directory.
        max_display: Features per waterfall.

    Returns:
        One record per explained patient, with its top contributing features.
    """
    import matplotlib.pyplot as plt
    import shap

    out = ensure_dir(out_dir)
    records: list[dict[str, Any]] = []

    for position, (index, label) in enumerate(zip(indices, labels), start=1):
        plt.figure()
        shap.plots.waterfall(explanation[int(index)], max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle(f"Patient row {index} — {label}", y=1.02, fontsize=11)
        path = out / f"shap_local_{position:02d}_{label}_row{index}.png"
        save_figure(fig, path)

        values = np.asarray(explanation.values)[int(index)]
        order = np.argsort(np.abs(values))[::-1][:5]
        records.append({
            "row": int(index),
            "case_type": label,
            "base_value": round(float(np.asarray(explanation.base_values)[int(index)]), 5),
            "shap_sum": round(float(values.sum()), 5),
            "top_features": [
                {"feature": explanation.feature_names[i],
                 "value": round(float(np.asarray(explanation.data)[int(index), i]), 4),
                 "shap": round(float(values[i]), 5)}
                for i in order
            ],
            "figure": path.name,
        })

    logger.info("Wrote %d local SHAP explanations to %s", len(records), out)
    return records


def run_shap_analysis(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    cfg: Config,
    out_dir: Path | str,
    case_selection: dict[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Run the full SHAP analysis: global ranking plus selected local explanations.

    Args:
        model: The selected fitted model (the base estimator, not the calibrator —
            calibration is a monotone transform of the output and does not change
            which features drove the decision).
        X_background: Training matrix, used as the reference distribution.
        X_explain: The matrix to explain, normally the test split.
        feature_names: Transformed feature names.
        cfg: Clinical configuration.
        out_dir: ``results/clinical/shap``.
        case_selection: ``{"TP": [rows], "FN": [rows], ...}`` chosen by the error
            analysis, so the local explanations cover both successes and failures.

    Returns:
        A summary dict, also written to ``shap_summary.json``.
    """
    out = ensure_dir(out_dir)
    max_display = int(cfg.get("explainability.shap.max_display", 15))
    n_local = int(cfg.get("explainability.shap.n_local_examples", 6))

    explainer = build_explainer(model, X_background, feature_names)
    explanation = _to_explanation(explainer, X_explain, feature_names)
    logger.info("Computed SHAP values: %s", np.asarray(explanation.values).shape)

    ranking = explain_global(explanation, out, max_display=max_display)

    # Pick local cases: spread across the requested categories, capped at n_local.
    indices: list[int] = []
    labels: list[str] = []
    if case_selection:
        per_category = max(1, n_local // max(len(case_selection), 1))
        for label, rows in case_selection.items():
            for row in list(rows)[:per_category]:
                indices.append(int(row))
                labels.append(label)
    if not indices:
        indices = list(range(min(n_local, X_explain.shape[0])))
        labels = ["case"] * len(indices)

    local = explain_local(explanation, indices[:n_local], labels[:n_local], out,
                          max_display=min(max_display, 12))

    summary = {
        "explainer": type(explainer).__name__,
        "model": type(model).__name__,
        "n_explained": int(X_explain.shape[0]),
        "n_features": int(len(feature_names)),
        "global_importance": ranking.to_dict(orient="records"),
        "local_explanations": local,
        "caveat": _CAVEAT,
    }
    save_json(summary, out / "shap_summary.json")
    return summary
