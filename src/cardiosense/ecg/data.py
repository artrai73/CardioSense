"""PTB-XL metadata handling, label construction and splitting.

This module knows the shape of PTB-XL and nothing else does. It answers four
questions:

1. Where are the files? (:func:`resolve_ptbxl_root`, :func:`verify_dataset`)
2. What are the labels? (:func:`build_superclass_labels`)
3. Which records go in which split? (:func:`split_by_fold`)
4. What is in this dataset anyway? (:func:`describe_dataset`)

PTB-XL structure, for reference::

    ptbxl/
    ├── ptbxl_database.csv    one row per record, indexed by ecg_id
    │                         key columns: patient_id, scp_codes, strat_fold,
    │                                      filename_lr (100 Hz), filename_hr (500 Hz)
    ├── scp_statements.csv    indexed by SCP code; `diagnostic` flags diagnostic
    │                         statements, `diagnostic_class` gives the superclass
    ├── records100/           100 Hz WFDB waveforms, nested 00000/, 01000/, ...
    └── records500/           500 Hz WFDB waveforms

``scp_codes`` is stored as a *stringified Python dict*, e.g. ``{'NORM': 100.0,
'SR': 0.0}``, mapping SCP code to a likelihood in 0-100. A likelihood of ``0.0``
means the likelihood was **not stated**, not that the statement is absent — so it
counts as present. This follows the convention in the PTB-XL benchmarking paper;
getting it wrong silently deletes a large fraction of the positive labels.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.paths import PATHS

__all__ = [
    "SUPERCLASSES",
    "resolve_ptbxl_root",
    "verify_dataset",
    "load_metadata",
    "parse_scp_codes",
    "build_superclass_labels",
    "split_by_fold",
    "describe_dataset",
]

logger = get_logger("ecg.data")

#: The five diagnostic superclasses, in a fixed order. This order defines the
#: column order of every label matrix, probability array and metric table in the
#: pipeline, and is written into the saved model metadata so inference cannot
#: silently permute them.
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def resolve_ptbxl_root(cfg: Config) -> Path:
    """Resolve the PTB-XL root directory from config, honouring the data root.

    ``dataset.root`` is written as ``data/ecg/ptbxl`` in the config. When
    ``$CARDIOSENSE_DATA_ROOT`` points at Google Drive, the ``data/`` prefix is
    rebased onto it, so nothing needs editing between machines.
    """
    value = str(cfg.dataset.root)
    if value.startswith("data/"):
        return (PATHS.data / value[len("data/"):]).resolve()
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PATHS.root / path).resolve()


def verify_dataset(cfg: Config, root: Path | None = None) -> dict[str, Any]:
    """Check that PTB-XL is present and self-consistent before any training starts.

    Args:
        cfg: ECG configuration.
        root: Override the resolved dataset root.

    Returns:
        A report describing what was found.

    Raises:
        FileNotFoundError: With download instructions, if the dataset is absent.
    """
    root = root or resolve_ptbxl_root(cfg)
    sampling_rate = int(cfg.dataset.sampling_rate)
    records_dir = root / f"records{sampling_rate}"
    database = root / str(cfg.dataset.database_csv)
    statements = root / str(cfg.dataset.scp_statements_csv)

    if not root.exists():
        raise FileNotFoundError(
            f"PTB-XL not found at {root}\n\n"
            "It is open access — no credentialing required. Download with:\n"
            "  wget https://physionet.org/static/published-projects/ptb-xl/"
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip\n"
            "then unzip it and move the extracted folder to the path above.\n"
            "See docs/datasets.md for the full instructions."
        )

    missing = [p.name for p in (database, statements, records_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"PTB-XL at {root} is incomplete. Missing: {missing}\n"
            f"Expected {database.name}, {statements.name} and records{sampling_rate}/ "
            "directly inside that folder — check you did not leave them one level deeper."
        )

    report = {
        "root": str(root),
        "sampling_rate": sampling_rate,
        "database_csv": str(database),
        "record_folders": len([p for p in records_dir.iterdir() if p.is_dir()]),
    }
    logger.info("PTB-XL verified at %s (%d record folders at %d Hz)",
                root, report["record_folders"], sampling_rate)
    return report


def load_metadata(cfg: Config, root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``ptbxl_database.csv`` and ``scp_statements.csv``.

    Returns:
        ``(database, scp_statements)``, both indexed as PTB-XL ships them.
    """
    root = root or resolve_ptbxl_root(cfg)
    database = pd.read_csv(root / str(cfg.dataset.database_csv), index_col="ecg_id")
    statements = pd.read_csv(root / str(cfg.dataset.scp_statements_csv), index_col=0)

    logger.info("Loaded metadata: %d records, %d patients, %d SCP statements",
                len(database), database.patient_id.nunique(), len(statements))
    return database, statements


def parse_scp_codes(value: Any) -> dict[str, float]:
    """Parse the stringified ``scp_codes`` dict into a real dict.

    ``ast.literal_eval`` is used rather than ``eval``: the file is data, and data
    should never be executed.
    """
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def build_superclass_labels(
    database: pd.DataFrame,
    statements: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Turn SCP codes into a 5-column multi-hot diagnostic superclass matrix.

    The mapping is: for every SCP code attached to a record, keep it if
    ``scp_statements.diagnostic == 1``, look up its ``diagnostic_class``, and set
    that superclass to 1.

    A likelihood of ``0.0`` means "likelihood not stated" and counts as present.
    Only codes with likelihood ``>= dataset.min_scp_likelihood`` (excluding the
    special 0) are kept, so raising that threshold filters to confident
    statements only.

    Records that end up with no superclass at all carry no usable supervision for
    this task; they are dropped when ``task.drop_records_without_label`` is set.

    Args:
        database: ``ptbxl_database.csv``.
        statements: ``scp_statements.csv``.
        cfg: ECG configuration.

    Returns:
        ``(filtered_database, label_matrix, report)`` where ``label_matrix`` has
        shape ``(n_records, 5)`` with columns ordered as :data:`SUPERCLASSES`.
    """
    classes = list(cfg.task.classes)
    if tuple(classes) != SUPERCLASSES:
        logger.warning("Config class order %s differs from the module default %s; "
                       "using the config order.", classes, list(SUPERCLASSES))

    min_likelihood = float(cfg.task.get("min_scp_likelihood", 0))

    diagnostic = statements[statements.diagnostic == 1]
    code_to_class = diagnostic.diagnostic_class.dropna().to_dict()
    logger.info("%d diagnostic SCP codes map onto %d superclasses",
                len(code_to_class), len(set(code_to_class.values())))

    class_index = {name: i for i, name in enumerate(classes)}
    labels = np.zeros((len(database), len(classes)), dtype=np.float32)
    unmapped: set[str] = set()

    for row, (_ecg_id, codes) in enumerate(database.scp_codes.items()):
        for code, likelihood in parse_scp_codes(codes).items():
            if code not in code_to_class:
                if code not in diagnostic.index:
                    continue          # non-diagnostic statement (rhythm/form): expected
                unmapped.add(code)
                continue
            # 0.0 means "likelihood not stated" and counts as present.
            if likelihood != 0.0 and likelihood < min_likelihood:
                continue
            superclass = code_to_class[code]
            if superclass in class_index:
                labels[row, class_index[superclass]] = 1.0

    has_label = labels.sum(axis=1) > 0
    report: dict[str, Any] = {
        "n_records_in": int(len(database)),
        "min_scp_likelihood": min_likelihood,
        "classes": classes,
        "records_without_any_superclass": int((~has_label).sum()),
        "diagnostic_codes_used": len(code_to_class),
        "unmapped_diagnostic_codes": sorted(unmapped),
    }

    if bool(cfg.task.get("drop_records_without_label", True)):
        database = database[has_label].copy()
        labels = labels[has_label]
        logger.info("Dropped %d records with no diagnostic superclass.",
                    report["records_without_any_superclass"])

    report.update({
        "n_records_out": int(len(database)),
        "n_patients": int(database.patient_id.nunique()),
        "prevalence": {name: round(float(labels[:, i].mean()), 4)
                       for i, name in enumerate(classes)},
        "support": {name: int(labels[:, i].sum()) for i, name in enumerate(classes)},
        "labels_per_record": {
            "mean": round(float(labels.sum(axis=1).mean()), 3),
            "max": int(labels.sum(axis=1).max()) if len(labels) else 0,
            "distribution": {str(int(k)): int(v) for k, v in
                             pd.Series(labels.sum(axis=1)).value_counts().sort_index().items()},
        },
    })

    logger.info("Labels built: %d records, prevalence %s",
                len(database), report["prevalence"])
    if report["labels_per_record"]["max"] > 1:
        logger.info(
            "%.1f%% of records carry more than one superclass — this is why the task is "
            "multi-label and why plain accuracy is not reported.",
            100 * float((labels.sum(axis=1) > 1).mean()),
        )
    return database, labels, report


def split_by_fold(
    database: pd.DataFrame,
    labels: np.ndarray,
    cfg: Config,
) -> dict[str, dict[str, Any]]:
    """Split using PTB-XL's official ``strat_fold`` column.

    Folds 1-10 were assigned by the dataset authors so that they are stratified by
    diagnostic class **and patient-disjoint**. Using them gives a leakage-free
    split for free and makes the results directly comparable to published work.

    Never re-split PTB-XL randomly by record: patients contribute several ECGs, so
    a record-level shuffle puts the same patient in train and test.

    Args:
        database: Filtered database (rows aligned with ``labels``).
        labels: Multi-hot label matrix.
        cfg: ECG configuration.

    Returns:
        ``{"train"/"val"/"test": {"indices", "database", "labels", ...}}``.

    Raises:
        ValueError: If the configured folds overlap or a split ends up empty.
    """
    train_folds = set(int(f) for f in cfg.split.train_folds)
    val_folds = set(int(f) for f in cfg.split.val_folds)
    test_folds = set(int(f) for f in cfg.split.test_folds)

    for a, b, name in ((train_folds, val_folds, "train/val"),
                       (train_folds, test_folds, "train/test"),
                       (val_folds, test_folds, "val/test")):
        if a & b:
            raise ValueError(f"Fold overlap between {name}: {sorted(a & b)}")

    fold = database.strat_fold.to_numpy()
    splits: dict[str, dict[str, Any]] = {}
    for name, folds in (("train", train_folds), ("val", val_folds), ("test", test_folds)):
        mask = np.isin(fold, list(folds))
        if not mask.any():
            raise ValueError(f"Split '{name}' is empty for folds {sorted(folds)}.")
        splits[name] = {
            "folds": sorted(folds),
            "indices": np.flatnonzero(mask),
            "database": database[mask],
            "labels": labels[mask],
        }

    # Patient disjointness is guaranteed by the dataset design, but assert it —
    # a silent violation would invalidate every number in the report.
    patients = {name: set(block["database"].patient_id) for name, block in splits.items()}
    overlaps = {
        "train_val": len(patients["train"] & patients["val"]),
        "train_test": len(patients["train"] & patients["test"]),
        "val_test": len(patients["val"] & patients["test"]),
    }
    if sum(overlaps.values()) > 0:
        raise ValueError(f"Patient overlap between splits: {overlaps}. This is leakage.")

    classes = list(cfg.task.classes)
    for name, block in splits.items():
        block["summary"] = {
            "folds": block["folds"],
            "n_records": int(len(block["indices"])),
            "n_patients": int(block["database"].patient_id.nunique()),
            "prevalence": {c: round(float(block["labels"][:, i].mean()), 4)
                           for i, c in enumerate(classes)},
            "support": {c: int(block["labels"][:, i].sum()) for i, c in enumerate(classes)},
        }
        logger.info("%-5s folds=%s records=%5d patients=%5d",
                    name, block["folds"], block["summary"]["n_records"],
                    block["summary"]["n_patients"])

    splits["patient_overlap"] = overlaps  # type: ignore[assignment]
    logger.info("Patient overlap between splits: %s (all zero = leakage-free)", overlaps)
    return splits


def describe_dataset(
    database: pd.DataFrame,
    labels: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Summary statistics used by the EDA section of the notebook."""
    classes = list(cfg.task.classes)
    co_occurrence = pd.DataFrame(
        labels.T @ labels, index=classes, columns=classes
    ).astype(int)

    return {
        "n_records": int(len(database)),
        "n_patients": int(database.patient_id.nunique()),
        "records_per_patient": {
            "mean": round(float(len(database) / max(database.patient_id.nunique(), 1)), 3),
            "max": int(database.patient_id.value_counts().max()),
        },
        "age": {
            "median": float(database.age.median()) if "age" in database else None,
            "iqr": [float(database.age.quantile(0.25)), float(database.age.quantile(0.75))]
            if "age" in database else None,
        },
        "sex_distribution": (
            {str(k): int(v) for k, v in database.sex.value_counts().items()}
            if "sex" in database else {}
        ),
        "class_support": {c: int(labels[:, i].sum()) for i, c in enumerate(classes)},
        "class_prevalence": {c: round(float(labels[:, i].mean()), 4)
                             for i, c in enumerate(classes)},
        "co_occurrence": co_occurrence.to_dict(),
        "multi_label_fraction": round(float((labels.sum(axis=1) > 1).mean()), 4),
    }
