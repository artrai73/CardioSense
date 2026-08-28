"""NIH ChestX-ray14 metadata handling, target extraction and patient-level splitting.

The single most important function here is :func:`split_by_patient`. A patient in
ChestX-ray14 can contribute dozens of films. Splitting by *image* puts the same
chest — same body habitus, same implanted hardware, same scanner — on both sides
of the split, and inflates AUC by several points. This is the best-documented
failure mode in the chest X-ray literature, and it is invisible in the metrics:
the model simply looks better than it is.

So every split here is by ``Patient ID``, and the disjointness is **asserted**,
not assumed.

Dataset structure::

    nih/
    ├── Data_Entry_2017_v2020.csv   one row per image
    │      Image Index | Finding Labels | Patient ID | View Position | ...
    ├── train_val_list.txt          official patient-disjoint train/val images
    ├── test_list.txt               official patient-disjoint test images
    └── images/                     112,120 flat PNGs, 1024x1024 grayscale

``Finding Labels`` is a pipe-separated string, e.g. ``Cardiomegaly|Effusion`` or
``No Finding``. The binary target is derived from it by substring test — no label
is invented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.paths import PATHS

__all__ = [
    "resolve_nih_root",
    "verify_dataset",
    "load_metadata",
    "build_target",
    "split_by_patient",
    "describe_dataset",
]

logger = get_logger("xray.data")

#: The Kaggle mirror ships `Data_Entry_2017.csv`; the NIH v2020 release renamed it.
#: Both are accepted so neither download route needs a config edit.
_METADATA_ALIASES = ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv")


def resolve_nih_root(cfg: Config) -> Path:
    """Resolve the dataset root, rebasing ``data/...`` onto ``$CARDIOSENSE_DATA_ROOT``."""
    value = str(cfg.dataset.root)
    if value.startswith("data/"):
        return (PATHS.data / value[len("data/"):]).resolve()
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PATHS.root / path).resolve()


def _find_metadata(root: Path, configured: str) -> Path:
    """Locate the label CSV, accepting either of the two shipped filenames."""
    candidates = [root / configured, *(root / name for name in _METADATA_ALIASES)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No label CSV found in {root}. Looked for: "
        f"{[c.name for c in candidates]}\n"
        "The Kaggle mirror names it Data_Entry_2017.csv; the NIH v2020 release uses "
        "Data_Entry_2017_v2020.csv. Either is fine — see docs/datasets.md."
    )


def verify_dataset(cfg: Config, root: Path | None = None) -> dict[str, Any]:
    """Check that ChestX-ray14 is present before any training starts.

    Raises:
        FileNotFoundError: With download instructions, if anything is missing.
    """
    root = root or resolve_nih_root(cfg)
    if not root.exists():
        raise FileNotFoundError(
            f"NIH ChestX-ray14 not found at {root}\n\n"
            "Download (Kaggle mirror, ~45 GB):\n"
            "  kaggle datasets download -d nih-chest-xrays/data -p /content/nih_raw --unzip\n"
            "then flatten images_001..012/images/ into a single images/ folder.\n"
            "See docs/datasets.md for the exact commands."
        )

    metadata_path = _find_metadata(root, str(cfg.dataset.metadata_csv))
    images_dir = root / str(cfg.dataset.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Image folder not found: {images_dir}\n"
            "The Kaggle mirror ships images in twelve folders (images_001/images/ ...). "
            "They must be flattened into one images/ directory — see docs/datasets.md."
        )

    n_images = sum(1 for _ in images_dir.glob("*.png"))
    report = {
        "root": str(root),
        "metadata_csv": str(metadata_path),
        "images_dir": str(images_dir),
        "n_image_files": n_images,
        "has_official_lists": (root / str(cfg.dataset.official_test_list)).exists()
        and (root / str(cfg.dataset.official_trainval_list)).exists(),
    }

    if n_images == 0:
        raise FileNotFoundError(f"{images_dir} contains no .png files.")
    logger.info("ChestX-ray14 verified at %s (%d images, official lists: %s)",
                root, n_images, report["has_official_lists"])
    if not report["has_official_lists"]:
        logger.warning(
            "Official train_val_list.txt / test_list.txt are missing. The pipeline will "
            "fall back to a grouped random split by patient, which is still leakage-free "
            "but no longer comparable to published splits."
        )
    return report


def load_metadata(cfg: Config, root: Path | None = None) -> pd.DataFrame:
    """Load the label CSV."""
    root = root or resolve_nih_root(cfg)
    metadata_path = _find_metadata(root, str(cfg.dataset.metadata_csv))
    frame = pd.read_csv(metadata_path)
    logger.info("Loaded %d rows from %s (%d patients)",
                len(frame), metadata_path.name, frame[cfg.dataset.patient_column].nunique())
    return frame


def build_target(
    frame: pd.DataFrame,
    cfg: Config,
    rng_seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive the binary target, filter views, and optionally subsample negatives.

    Steps:

    1. **Target.** Positive iff the configured label appears in ``Finding Labels``.
       Nothing is invented — the string is already in the released metadata.
    2. **View filter.** PA films only, by default. AP films are portable studies
       on supine, sicker patients and geometrically magnify the cardiac
       silhouette. Mixing views hands the model a shortcut: "predict cardiomegaly
       whenever the film looks like a portable AP", which has nothing to do with
       heart size.
    3. **Negative subsampling.** Optional, and done **by patient** so the
       downstream split stays patient-disjoint. This changes the prevalence, so
       the sampling ratio is recorded in the metadata and every emitted
       probability is conditioned on the sampled prior.

    Args:
        frame: Raw metadata.
        cfg: X-ray configuration.
        rng_seed: Seed for subsampling; defaults to ``cfg.seed``.

    Returns:
        ``(filtered_frame, report)``. The frame gains a ``target`` column.
    """
    dataset = cfg.dataset
    target_label = str(dataset.target_label)
    label_column = str(dataset.label_column)
    patient_column = str(dataset.patient_column)

    df = frame.copy()
    report: dict[str, Any] = {"rows_in": int(len(df)), "target_label": target_label}

    missing = [c for c in (label_column, patient_column, str(dataset.image_column))
               if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from the metadata: {missing}. "
                       f"Available: {list(df.columns)}")

    # -- 1. target ---------------------------------------------------------
    df["target"] = df[label_column].fillna("").str.contains(
        target_label, regex=False
    ).astype(int)
    report["prevalence_all_views"] = round(float(df.target.mean()), 5)
    report["positives_all_views"] = int(df.target.sum())

    # -- 2. view filter ----------------------------------------------------
    view_filter = dataset.get("filter_view")
    view_column = str(dataset.get("view_column", "View Position"))
    if view_filter and view_column in df.columns:
        before = len(df)
        df = df[df[view_column] == view_filter].copy()
        report["view_filter"] = view_filter
        report["rows_after_view_filter"] = int(len(df))
        logger.info("View filter %s: %d -> %d images", view_filter, before, len(df))

    report["prevalence_after_view_filter"] = round(float(df.target.mean()), 5)

    # -- 3. negative subsampling, BY PATIENT -------------------------------
    ratio = dataset.get("negative_ratio")
    if ratio:
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(cfg.seed))
        positive_patients = set(df.loc[df.target == 1, patient_column])
        negative_only = df[~df[patient_column].isin(positive_patients)]
        negative_patients = negative_only[patient_column].unique()

        n_positive_images = int(df.target.sum())
        target_negatives = int(ratio) * n_positive_images

        # Sample whole patients, not images, so the split stays patient-disjoint.
        images_per_patient = max(len(negative_only) / max(len(negative_patients), 1), 1.0)
        n_patients_needed = min(len(negative_patients),
                                int(np.ceil(target_negatives / images_per_patient)))
        keep_patients = set(rng.choice(negative_patients, size=n_patients_needed,
                                       replace=False)) if n_patients_needed else set()

        before = len(df)
        df = df[df[patient_column].isin(positive_patients | keep_patients)].copy()
        report["negative_subsampling"] = {
            "requested_ratio": int(ratio),
            "rows_before": int(before),
            "rows_after": int(len(df)),
            "negative_patients_kept": len(keep_patients),
            "sampled_by": "patient",
            "note": "Subsampling changes the prevalence. Emitted probabilities are "
                    "conditioned on this sampled prior and need correcting before they "
                    "are read as population risk.",
        }
        logger.info("Negative subsampling (ratio %s, by patient): %d -> %d images",
                    ratio, before, len(df))

    max_images = dataset.get("max_images")
    if max_images and len(df) > int(max_images):
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(cfg.seed))
        keep_patients = df[patient_column].unique()
        rng.shuffle(keep_patients)
        selected: list[Any] = []
        running = 0
        counts = df[patient_column].value_counts()
        for patient in keep_patients:
            if running >= int(max_images):
                break
            selected.append(patient)
            running += int(counts[patient])
        df = df[df[patient_column].isin(selected)].copy()
        report["max_images_cap"] = {"cap": int(max_images), "rows_after": int(len(df))}
        logger.info("Capped to %d images (%d patients) for a fast run.",
                    len(df), len(selected))

    report.update({
        "rows_out": int(len(df)),
        "n_patients": int(df[patient_column].nunique()),
        "positives": int(df.target.sum()),
        "negatives": int((df.target == 0).sum()),
        "final_prevalence": round(float(df.target.mean()), 5),
        "images_per_patient": round(float(len(df) / max(df[patient_column].nunique(), 1)), 3),
    })

    logger.info("Target '%s': %d positives of %d images (%.2f%%) across %d patients",
                target_label, report["positives"], report["rows_out"],
                100 * report["final_prevalence"], report["n_patients"])

    if report["final_prevalence"] < 0.15:
        logger.warning(
            "Prevalence is %.2f%%. A model predicting 'negative' for everything scores "
            "%.1f%% accuracy and is clinically worthless — PR-AUC is the headline metric, "
            "not accuracy.",
            100 * report["final_prevalence"], 100 * (1 - report["final_prevalence"]),
        )
    return df, report


def _read_official_list(path: Path) -> set[str]:
    """Read one of NIH's newline-delimited image lists."""
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def split_by_patient(
    frame: pd.DataFrame,
    cfg: Config,
    root: Path | None = None,
) -> dict[str, Any]:
    """Split into train / val / test **by patient**, never by image.

    Two strategies:

    * ``official`` — use NIH's own patient-disjoint ``train_val_list.txt`` and
      ``test_list.txt``, then carve validation out of the train-val portion **by
      patient**. This keeps the test split comparable to published work.
    * ``grouped_random`` — a patient-level random split, used when the official
      lists are unavailable.

    Either way the function asserts that no patient appears in two splits, and
    records the check in the returned summary so it is visible in the results
    rather than merely believed.

    Args:
        frame: Filtered metadata with a ``target`` column.
        cfg: X-ray configuration.
        root: Dataset root, for locating the official lists.

    Returns:
        ``{"train"/"val"/"test": {...}, "summary": {...}}``.

    Raises:
        ValueError: If any split is empty or patient overlap is detected.
    """
    root = root or resolve_nih_root(cfg)
    patient_column = str(cfg.dataset.patient_column)
    image_column = str(cfg.dataset.image_column)
    strategy = str(cfg.split.get("strategy", "official")).lower()
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)

    official_available = (
        (root / str(cfg.dataset.official_test_list)).exists()
        and (root / str(cfg.dataset.official_trainval_list)).exists()
    )
    if strategy == "official" and not official_available:
        logger.warning("Official lists unavailable; falling back to grouped_random.")
        strategy = "grouped_random"

    if strategy == "official":
        test_images = _read_official_list(root / str(cfg.dataset.official_test_list))
        is_test = frame[image_column].isin(test_images)
        test_frame = frame[is_test]
        trainval_frame = frame[~is_test]

        # NIH's lists are patient-disjoint, but a patient could in principle
        # appear in both after our filtering. Enforce it explicitly.
        test_patients = set(test_frame[patient_column])
        trainval_frame = trainval_frame[~trainval_frame[patient_column].isin(test_patients)]
    else:
        patients = frame[patient_column].unique()
        rng.shuffle(patients)
        n_test = max(1, int(len(patients) * float(cfg.split.get("test_size", 0.15))))
        test_patients = set(patients[:n_test])
        test_frame = frame[frame[patient_column].isin(test_patients)]
        trainval_frame = frame[~frame[patient_column].isin(test_patients)]

    # Validation is carved out of train-val BY PATIENT.
    trainval_patients = trainval_frame[patient_column].unique().copy()
    rng.shuffle(trainval_patients)
    n_val = max(1, int(len(trainval_patients) * float(cfg.split.get("val_size", 0.15))))
    val_patients = set(trainval_patients[:n_val])

    val_frame = trainval_frame[trainval_frame[patient_column].isin(val_patients)]
    train_frame = trainval_frame[~trainval_frame[patient_column].isin(val_patients)]

    splits = {"train": train_frame, "val": val_frame, "test": test_frame}

    for name, block in splits.items():
        if len(block) == 0:
            raise ValueError(f"Split '{name}' is empty. Check the filters and split sizes.")

    patients = {name: set(block[patient_column]) for name, block in splits.items()}
    overlap = {
        "train_val": len(patients["train"] & patients["val"]),
        "train_test": len(patients["train"] & patients["test"]),
        "val_test": len(patients["val"] & patients["test"]),
    }
    if sum(overlap.values()) > 0:
        raise ValueError(
            f"PATIENT OVERLAP between splits: {overlap}. This is leakage — the same "
            "chest would appear in training and evaluation."
        )

    summary: dict[str, Any] = {
        "strategy": strategy,
        "seed": seed,
        "patient_overlap": overlap,
        "splits": {},
    }
    for name, block in splits.items():
        summary["splits"][name] = {
            "n_images": int(len(block)),
            "n_patients": int(block[patient_column].nunique()),
            "n_positive": int(block.target.sum()),
            "prevalence": round(float(block.target.mean()), 5),
            "images_per_patient": round(
                float(len(block) / max(block[patient_column].nunique(), 1)), 2),
        }
        logger.info("%-5s %6d images | %5d patients | %4d positive (%.2f%%)",
                    name, len(block), block[patient_column].nunique(),
                    int(block.target.sum()), 100 * block.target.mean())

    logger.info("Patient overlap between splits: %s (all zero = leakage-free)", overlap)

    prevalences = [summary["splits"][n]["prevalence"] for n in ("train", "val", "test")]
    if max(prevalences) - min(prevalences) > 0.10:
        logger.warning(
            "Prevalence differs by %.1f points across splits (%s). Patient-level "
            "splitting cannot stratify perfectly when patients contribute different "
            "numbers of images; note this when comparing split metrics.",
            100 * (max(prevalences) - min(prevalences)),
            {n: summary["splits"][n]["prevalence"] for n in ("train", "val", "test")},
        )

    return {"train": train_frame, "val": val_frame, "test": test_frame, "summary": summary}


def describe_dataset(frame: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Summary statistics for the EDA section."""
    patient_column = str(cfg.dataset.patient_column)
    label_column = str(cfg.dataset.label_column)

    all_findings: dict[str, int] = {}
    for entry in frame[label_column].fillna(""):
        for finding in str(entry).split("|"):
            finding = finding.strip()
            if finding:
                all_findings[finding] = all_findings.get(finding, 0) + 1

    positive = frame[frame.target == 1]
    co_occurring: dict[str, int] = {}
    for entry in positive[label_column].fillna(""):
        for finding in str(entry).split("|"):
            finding = finding.strip()
            if finding and finding != str(cfg.dataset.target_label):
                co_occurring[finding] = co_occurring.get(finding, 0) + 1

    description: dict[str, Any] = {
        "n_images": int(len(frame)),
        "n_patients": int(frame[patient_column].nunique()),
        "prevalence": round(float(frame.target.mean()), 5),
        "finding_counts": dict(sorted(all_findings.items(), key=lambda kv: -kv[1])),
        "co_occurring_with_target": dict(sorted(co_occurring.items(), key=lambda kv: -kv[1])),
        "images_per_patient": {
            "mean": round(float(len(frame) / max(frame[patient_column].nunique(), 1)), 2),
            "max": int(frame[patient_column].value_counts().max()),
        },
    }

    for column, key in (("Patient Age", "age"), ("Patient Gender", "sex"),
                        ("View Position", "view")):
        if column in frame.columns:
            if key == "age":
                ages = pd.to_numeric(frame[column], errors="coerce")
                ages = ages[(ages > 0) & (ages < 120)]   # NIH has some corrupt ages
                description["age"] = {"median": float(ages.median()),
                                      "iqr": [float(ages.quantile(0.25)),
                                              float(ages.quantile(0.75))]}
            else:
                description[key] = {str(k): int(v)
                                    for k, v in frame[column].value_counts().items()}
    return description
