"""Baselines for the chest X-ray task (Experiment X-A).

Two references, both cheap, and each answering a different question:

**Majority class.** Predicts "no cardiomegaly" for every image. Its only purpose
is to make the accuracy trap visible: at ~2.5% prevalence this scores ~97.5%
accuracy while never identifying a single case. Reporting it in the comparison
table means nobody reading the report can mistake a high accuracy for a working
model. Its PR-AUC equals the prevalence, which is the honest chance level.

**Logistic regression on simple image features.** Downsampled pixels plus an
intensity histogram, fed to a linear model. This is the floor DenseNet121 must
clear. If a 32x32 thumbnail and a grey-level histogram get most of the way to the
CNN's performance, then the CNN is largely reading global exposure and body size
rather than cardiac silhouette shape, and the result deserves scepticism.

Features
--------

======================  ==============================================
Feature block           What it captures
======================  ==============================================
Downsampled pixels      Coarse anatomy: mediastinal width, lung field
(32x32 = 1024 values)   extent, overall shape
Intensity histogram     Exposure, contrast and tissue-density
(32 bins)               distribution, independent of position
======================  ==============================================

Deliberately crude. A baseline that needs tuning is not a baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.metrics import binary_metrics

__all__ = ["extract_image_features", "build_feature_matrix", "majority_class_baseline",
           "train_pixel_baseline"]

logger = get_logger("xray.baseline")


def extract_image_features(
    path: Path | str,
    downsample_size: int = 32,
    hist_bins: int = 32,
) -> np.ndarray:
    """Extract simple features from one image.

    Args:
        path: Image file.
        downsample_size: Thumbnail edge length.
        hist_bins: Histogram bin count.

    Returns:
        A flat vector of length ``downsample_size**2 + hist_bins``.
    """
    from PIL import Image

    with Image.open(path) as handle:
        grey = handle.convert("L")
        thumbnail = grey.resize((downsample_size, downsample_size), Image.BILINEAR)
        pixels = np.asarray(thumbnail, dtype=np.float32) / 255.0
        full = np.asarray(grey, dtype=np.float32) / 255.0

    histogram, _edges = np.histogram(full, bins=hist_bins, range=(0.0, 1.0), density=True)
    return np.concatenate([pixels.ravel(), histogram.astype(np.float32)])


def build_feature_matrix(
    frame: pd.DataFrame,
    images_dir: Path | str,
    cfg: Config,
    show_progress: bool = True,
) -> np.ndarray:
    """Build the feature matrix for a split.

    Args:
        frame: Split metadata.
        images_dir: Directory holding the PNGs.
        cfg: X-ray configuration.
        show_progress: Show a progress bar.

    Returns:
        Shape ``(len(frame), n_features)``.
    """
    images_dir = Path(images_dir)
    image_column = str(cfg.dataset.image_column)
    downsample = int(cfg.baseline.get("downsample_size", 32))
    bins = int(cfg.baseline.get("hist_bins", 32))

    filenames: Sequence[str] = frame[image_column].tolist()
    iterator: Any = filenames
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(filenames, desc="extracting image features")

    rows = [extract_image_features(images_dir / name, downsample, bins) for name in iterator]
    matrix = np.vstack(rows).astype(np.float32)
    logger.info("Feature matrix: %s (%d downsampled pixels + %d histogram bins)",
                matrix.shape, downsample ** 2, bins)
    return matrix


def majority_class_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """The 'always predict the majority class' baseline.

    Exists to make the accuracy trap explicit in the results table. Its
    probability output is the constant training prevalence, so its PR-AUC equals
    the test prevalence — the true chance level for this metric.
    """
    y_train = np.asarray(y_train).ravel().astype(int)
    y_test = np.asarray(y_test).ravel().astype(int)

    majority = int(np.bincount(y_train, minlength=2).argmax())
    prevalence = float(y_train.mean())

    y_pred = np.full_like(y_test, majority)
    y_prob = np.full(y_test.shape, prevalence, dtype=float)

    metrics = binary_metrics(y_test, y_pred, y_prob)
    metrics["threshold"] = 0.5

    logger.info(
        "Majority-class baseline: accuracy %.4f, recall %.4f, PR-AUC %.4f. "
        "It identifies %d of %d positive cases — which is why accuracy is not the "
        "headline metric.",
        metrics["accuracy"], metrics["recall"], metrics.get("pr_auc", float("nan")),
        int(metrics["tp"]), int(y_test.sum()),
    )
    return {"metrics": metrics, "majority_class": majority,
            "constant_probability": round(prevalence, 5), "parameters": 1}


def train_pixel_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    cfg: Config,
    X_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Logistic regression on downsampled pixels and an intensity histogram.

    Standardisation is fitted on train only, the same discipline as everywhere
    else. ``class_weight="balanced"`` handles the imbalance, matching the
    ``pos_weight`` correction the deep model gets, so the comparison is fair.
    """
    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ..common.compat import logistic_penalty_kwargs

    params = cfg.baseline
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=int(params.get("max_iter", 1000)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=int(cfg.seed),
            **logistic_penalty_kwargs("l2"),
        )),
    ])

    logger.info("Fitting pixel-feature baseline: %d features, %d images",
                X_train.shape[1], X_train.shape[0])
    start = time.time()
    pipeline.fit(X_train, np.asarray(y_train).ravel().astype(int))
    elapsed = time.time() - start

    result: dict[str, Any] = {
        "model": pipeline,
        "n_features": int(X_train.shape[1]),
        "parameters": int(X_train.shape[1] + 1),
        "train_seconds": round(elapsed, 2),
        "val_prob": pipeline.predict_proba(X_val)[:, 1].astype(np.float32),
    }
    if X_test is not None:
        result["test_prob"] = pipeline.predict_proba(X_test)[:, 1].astype(np.float32)

    logger.info("Pixel baseline fitted in %.1fs", elapsed)
    return result
