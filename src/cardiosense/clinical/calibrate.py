"""Probability calibration for the clinical model.

Why this module exists at all
-----------------------------

Phase 2 fuses three modalities by weighting each one by how much it should be
trusted. That only works if the number each pipeline emits means what it says: if
the clinical model outputs 0.80 for a group of patients, roughly 80% of them
should actually have disease. A model can rank patients perfectly (ROC-AUC 0.95)
and still be badly wrong about the *level* of its probabilities. Gradient-boosted
trees in particular tend to push probabilities toward 0 and 1, because each
boosting round is optimising a loss that rewards confident correct answers.

Vocabulary used consistently across the codebase
------------------------------------------------

``prediction probability``
    The raw number the classifier emits. It is a **score**: monotone in risk, so
    it is valid for ranking, AUC and threshold selection. It carries no guarantee
    about frequency.

``calibrated confidence / reliability``
    The output after calibration, which is a **frequency claim**: among all
    patients assigned 0.80, about 80% are positive. This is what a clinician can
    reason with, and the only version that is legitimate to use as a fusion
    weight in Phase 2.

Both are reported. They are never conflated.

How the method is chosen without leaking
----------------------------------------

The final calibrator is fitted on the **validation** split, because the base model
never saw those patients. That creates a subtle problem: you cannot then *choose*
between Platt scaling and isotonic regression by scoring them on validation,
because both were fitted on it — isotonic, being non-parametric, will look almost
perfect there by construction. Choosing on test is straightforward leakage.

So the choice is made by cross-validation on the **training** split:

    for each of k folds:
        fit a fresh copy of the base model on the other k-1 folds
        fit each candidate calibrator on this held-out fold
        score each calibrator on that same held-out fold, out-of-model-sample

The winner by mean score is then fitted once on validation to produce the
deliverable calibrator. Train is used to pick the *method*, validation to fit the
*parameters*, test to report. No split does two jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from ..common.compat import make_prefit_calibrator
from ..common.config import Config
from ..common.io_utils import save_json
from ..common.logging_utils import get_logger
from ..common.metrics import brier_score, expected_calibration_error
from ..common.paths import ensure_dir
from ..common.plots import plot_calibration_curve

__all__ = ["select_calibration_method", "fit_calibrator", "calibration_report",
           "calibrated_confidence", "clip_probabilities", "PROBABILITY_CLIP"]

logger = get_logger(__name__)

_EPS = 1e-12

#: Calibrated probabilities are clipped into this range before being returned.
#: A calibrated probability of exactly 1.0 asserts certainty, and no calibrator
#: fitted on a few hundred patients can support that claim — isotonic regression
#: in particular saturates at the ends of its range. Clipping keeps the output an
#: honest confidence and keeps log-loss finite.
PROBABILITY_CLIP = (0.001, 0.999)


def clip_probabilities(probabilities: np.ndarray | float) -> np.ndarray | float:
    """Clip probabilities into :data:`PROBABILITY_CLIP`. See the note above."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), *PROBABILITY_CLIP)
    return float(clipped) if clipped.ndim == 0 else clipped


def _score(y_true: np.ndarray, y_prob: np.ndarray, metric: str, n_bins: int = 10) -> float:
    """Lower-is-better calibration score."""
    if metric == "brier":
        return brier_score(y_true, y_prob)
    if metric == "ece":
        return expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    if metric == "log_loss":
        return float(log_loss(y_true, np.clip(y_prob, _EPS, 1 - _EPS), labels=[0, 1]))
    raise ValueError(f"Unknown calibration metric {metric!r}; use brier, ece or log_loss.")


def _fit_1d_mapping(method: str, probabilities: np.ndarray, labels: np.ndarray) -> Any:
    """Fit a one-dimensional probability -> probability mapping.

    This is the mathematical core of both calibration methods, isolated so it can
    be cross-validated honestly:

    * ``sigmoid`` (Platt) — logistic regression on the log-odds of the input
      probability. Two parameters, so it is stable on small samples, and it can
      only stretch or shift the curve; it cannot invent structure.
    * ``isotonic`` — a non-decreasing step function. Far more flexible, and
      correspondingly far easier to overfit.

    Returns:
        A callable mapping raw probabilities to calibrated probabilities.
    """
    p = np.clip(np.asarray(probabilities, dtype=float).ravel(), _EPS, 1 - _EPS)
    y = np.asarray(labels).ravel().astype(int)

    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        z = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(max_iter=1000).fit(z, y)

        def apply_sigmoid(q: np.ndarray) -> np.ndarray:
            q = np.clip(np.asarray(q, dtype=float).ravel(), _EPS, 1 - _EPS)
            return model.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]

        return apply_sigmoid

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)

        def apply_isotonic(q: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(np.asarray(q, dtype=float).ravel()))

        return apply_isotonic

    raise ValueError(f"Unknown calibration method {method!r}; use sigmoid or isotonic.")


def select_calibration_method(
    base_estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> tuple[str, dict[str, Any]]:
    """Choose between Platt scaling and isotonic regression, without leaking.

    Two stages, and the separation between them is the whole point:

    1. **Out-of-fold base predictions.** ``cross_val_predict`` scores every
       training patient with a model that did not see them. This yields ~200
       honest (probability, outcome) pairs.
    2. **Inner cross-validation of the 1-D mapping.** Each candidate calibrator is
       fitted on a subset of those pairs and scored on the *held-out* remainder.

    Stage 2 is what makes the comparison fair. An earlier version of this function
    fitted the calibrator on a fold and scored it on **that same fold**, which
    isotonic regression wins automatically: being non-parametric, it can fit the
    fold almost exactly. On test data that apparent advantage evaporated and the
    Brier score got *worse*, while inference emitted calibrated probabilities of
    exactly 1.0 — a certainty claim no finite sample supports. Scoring on held-out
    pairs removes that artefact and lets the two-parameter sigmoid win when it
    deserves to, which on a few hundred patients is most of the time.

    Args:
        base_estimator: The tuned estimator; cloned, never mutated.
        X_train: Preprocessed training matrix.
        y_train: Training labels.
        cfg: Clinical configuration.

    Returns:
        ``(method_name, report)`` where method is ``"sigmoid"`` or ``"isotonic"``.
    """
    from sklearn.model_selection import cross_val_predict

    methods: Sequence[str] = list(cfg.calibration.get("methods", ["sigmoid", "isotonic"]))
    metric = str(cfg.calibration.get("selection_metric", "brier"))
    n_bins = int(cfg.calibration.get("n_bins", 10))
    folds = int(cfg.calibration.get("selection_cv_folds", 5))

    y_train = np.asarray(y_train).ravel().astype(int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg.seed))

    # -- stage 1: honest out-of-fold probabilities -------------------------
    oof = np.asarray(
        cross_val_predict(clone(base_estimator), X_train, y_train, cv=outer,
                          method="predict_proba", n_jobs=-1)
    )[:, 1]
    baseline = _score(y_train, oof, metric, n_bins)

    # -- stage 2: cross-validate each candidate mapping on those pairs -----
    inner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg.seed) + 1)
    scores: dict[str, list[float]] = {method: [] for method in methods}

    for fold, (fit_idx, eval_idx) in enumerate(inner.split(oof.reshape(-1, 1), y_train), start=1):
        for method in methods:
            try:
                mapping = _fit_1d_mapping(method, oof[fit_idx], y_train[fit_idx])
                scores[method].append(
                    _score(y_train[eval_idx], mapping(oof[eval_idx]), metric, n_bins)
                )
            except Exception as exc:  # noqa: BLE001 - degenerate folds are possible
                logger.warning("Fold %d: %s calibration failed (%s); scored as +inf.",
                               fold, method, exc)
                scores[method].append(float("inf"))

    means = {method: float(np.mean(values)) for method, values in scores.items()}
    stds = {method: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            for method, values in scores.items()}
    best = min(means, key=lambda m: means[m])

    report = {
        "selection_metric": metric,
        "procedure": "out-of-fold base predictions on train, then inner CV of the "
                     "1-D calibration mapping (fitted and scored on disjoint pairs)",
        "cv_folds": folds,
        "n_oof_pairs": int(len(oof)),
        "uncalibrated_oof": round(baseline, 5),
        "candidates": {
            method: {"mean": round(means[method], 5), "std": round(stds[method], 5),
                     "per_fold": [round(v, 5) for v in scores[method]]}
            for method in methods
        },
        "selected": best,
        "improvement_over_uncalibrated": round(baseline - means[best], 5),
    }

    logger.info("Calibration method selection (held-out %s on %d OOF pairs): %s",
                metric, len(oof), {k: round(v, 5) for k, v in means.items()})
    logger.info("Selected '%s' (uncalibrated %s = %.5f -> %.5f)",
                best, metric, baseline, means[best])

    if means[best] >= baseline:
        report["warning"] = (
            "No candidate improved on the uncalibrated model in cross-validation. The base "
            "model is already reasonably calibrated; calibration is retained anyway because "
            "it costs nothing at inference and stabilises the probabilities used for fusion."
        )
        logger.warning(report["warning"])

    if best == "isotonic" and len(oof) < 1000:
        report["note"] = (
            f"Isotonic won on only {len(oof)} calibration pairs. It is non-parametric and can "
            "saturate to exactly 0 or 1, which is never an honest confidence claim. Predicted "
            "probabilities are clipped at inference, and the test Brier score is compared "
            "against the uncalibrated baseline before this is trusted."
        )
        logger.warning(report["note"])

    return best, report


def fit_calibrator(
    fitted_model: Any,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    method: str = "sigmoid",
) -> Any:
    """Fit the deliverable calibrator on the VALIDATION split.

    The base model is frozen — ``make_prefit_calibrator`` wraps it so that
    scikit-learn calibrates the existing predictions rather than refitting the
    model on the calibration data. This is verified by the caller.

    Args:
        fitted_model: The already-trained, selected estimator.
        X_calibration: Preprocessed validation matrix.
        y_calibration: Validation labels.
        method: ``"sigmoid"`` (Platt) or ``"isotonic"``.

    Returns:
        A fitted calibrator exposing ``predict_proba``.
    """
    logger.info("Fitting %s calibrator on %d validation patients (base model frozen).",
                method, len(y_calibration))
    calibrator = make_prefit_calibrator(fitted_model, method=method)
    calibrator.fit(X_calibration, np.asarray(y_calibration).ravel())
    return calibrator


def calibrated_confidence(calibrated_probability: float | np.ndarray) -> float | np.ndarray:
    """Confidence in the returned decision, derived from the calibrated probability.

    Defined as ``max(p, 1 - p)``: a calibrated probability of 0.05 is a *confident*
    prediction of "no disease", while 0.52 is an uncertain prediction of "disease".
    This is the quantity Phase 2 fusion should weight by — not the probability
    itself, which conflates direction with strength.
    """
    p = np.asarray(calibrated_probability, dtype=float)
    confidence = np.maximum(p, 1.0 - p)
    return float(confidence) if confidence.ndim == 0 else confidence


def calibration_report(
    y_true: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    cfg: Config,
    out_dir: Path | str,
    method: str,
    split_name: str = "test",
) -> dict[str, Any]:
    """Compare uncalibrated and calibrated probabilities, and plot the reliability diagram.

    Reports Brier score, ECE and log loss for both, plus the mean predicted
    probability against the observed positive rate — the single clearest summary
    of whether a model is systematically over- or under-confident.

    Args:
        y_true: True labels for the split.
        raw_prob: Uncalibrated probabilities.
        calibrated_prob: Calibrated probabilities.
        cfg: Clinical configuration.
        out_dir: ``results/clinical``.
        method: Calibration method actually used.
        split_name: Which split these probabilities come from.

    Returns:
        The comparison dict, also written to ``calibration_metrics.json``.
    """
    out = ensure_dir(out_dir)
    y_true = np.asarray(y_true).ravel().astype(int)
    raw_prob = np.asarray(raw_prob).ravel()
    calibrated_prob = np.asarray(calibrated_prob).ravel()
    n_bins = int(cfg.calibration.get("n_bins", 10))

    def block(probs: np.ndarray) -> dict[str, float]:
        return {
            "brier": round(brier_score(y_true, probs), 5),
            "ece": round(expected_calibration_error(y_true, probs, n_bins=n_bins), 5),
            "ece_quantile_bins": round(
                expected_calibration_error(y_true, probs, n_bins=n_bins, strategy="quantile"), 5),
            "log_loss": round(float(log_loss(y_true, np.clip(probs, _EPS, 1 - _EPS),
                                             labels=[0, 1])), 5),
            "mean_predicted_probability": round(float(probs.mean()), 5),
        }

    before, after = block(raw_prob), block(calibrated_prob)
    observed = round(float(y_true.mean()), 5)

    # Bin count is capped so a 45-patient split is not sliced into empty bins.
    plot_bins = max(3, min(n_bins, len(y_true) // 6))
    plot_calibration_curve(
        {"Uncalibrated": (y_true, raw_prob), f"Calibrated ({method})": (y_true, calibrated_prob)},
        out / "calibration_curve.png",
        n_bins=plot_bins,
        strategy=str(cfg.calibration.get("curve_strategy", "uniform")),
        title=f"Reliability diagram — {split_name} split (n = {len(y_true)}, "
              f"{plot_bins} bins)",
    )

    report = {
        "split": split_name,
        "method": method,
        "n_samples": int(len(y_true)),
        "observed_positive_rate": observed,
        "uncalibrated": before,
        "calibrated": after,
        "improvement": {
            key: round(before[key] - after[key], 5)
            for key in ("brier", "ece", "log_loss")
        },
        "plot_bins": plot_bins,
        "interpretation": {
            "prediction_probability": "Raw classifier output. A score: valid for ranking, "
                                      "AUC and thresholding. Makes no frequency guarantee.",
            "calibrated_confidence": "Post-calibration output. A frequency claim: among "
                                     "patients assigned p, about p of them are positive. This "
                                     "is the value Phase 2 fusion should weight by.",
        },
    }

    logger.info("Calibration on %s — Brier %.5f -> %.5f | ECE %.5f -> %.5f",
                split_name, before["brier"], after["brier"], before["ece"], after["ece"])
    logger.info("Mean predicted probability %.3f (uncal) / %.3f (cal) vs observed rate %.3f",
                before["mean_predicted_probability"], after["mean_predicted_probability"],
                observed)

    if after["brier"] > before["brier"]:
        report["warning"] = (
            "Calibration made the Brier score worse on this split. With a small calibration "
            "set this is a real possibility, and it should be reported rather than hidden. "
            "Consider sigmoid over isotonic, or report the uncalibrated probabilities."
        )
        logger.warning(report["warning"])

    save_json(report, out / "calibration_metrics.json")
    return report
