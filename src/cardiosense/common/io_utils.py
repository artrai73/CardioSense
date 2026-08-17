"""Serialisation helpers.

Centralised so that (a) NumPy scalars never break a ``json.dump`` at the end of a
two-hour training run, and (b) every artifact lands in a predictable place with
its parent directory created.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_dataframe",
    "timestamp",
    "to_serializable",
    "NumpyJSONEncoder",
]


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """UTC timestamp string, used for experiment run IDs."""
    return datetime.now(timezone.utc).strftime(fmt)


def to_serializable(obj: Any) -> Any:
    """Convert NumPy / Path / datetime objects into JSON-friendly Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """``json`` encoder that understands NumPy scalars, arrays and ``Path``."""

    def default(self, o: Any) -> Any:  # noqa: D102
        converted = to_serializable(o)
        if converted is o:
            return super().default(o)
        return converted


def save_json(data: Any, path: Path | str, indent: int = 2) -> Path:
    """Write *data* to *path* as JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, indent=indent, cls=NumpyJSONEncoder)
    return target


def load_json(path: Path | str) -> Any:
    """Read JSON from *path*."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(obj: Any, path: Path | str, use_joblib: bool = True) -> Path:
    """Persist *obj*.

    ``joblib`` is preferred for scikit-learn estimators (it stores large NumPy
    arrays far more efficiently than plain pickle).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if use_joblib:
        try:
            import joblib

            joblib.dump(obj, target)
            return target
        except ImportError:
            pass
    with target.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_pickle(path: Path | str, use_joblib: bool = True) -> Any:
    """Load an object written by :func:`save_pickle`."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Artifact not found: {source}")
    if use_joblib:
        try:
            import joblib

            return joblib.load(source)
        except ImportError:
            pass
    with source.open("rb") as handle:
        return pickle.load(handle)


def save_dataframe(df: Any, path: Path | str, index: bool = False) -> Path:
    """Write a pandas DataFrame to CSV, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=index)
    return target
