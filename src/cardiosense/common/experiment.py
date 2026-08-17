"""Lightweight experiment tracking (Part 10 of the Phase 1 spec).

Deliberately not MLflow or Weights & Biases: those need a server or an account,
break when a Colab runtime dies, and add a dependency the examiners cannot run
offline. This module writes one self-contained JSON per run plus an append-only
CSV index, both of which live in Git and open in Excel.

Usage::

    with ExperimentTracker("xgboost_tuned", modality="clinical", config=cfg) as run:
        run.log_params({"n_estimators": 300, "max_depth": 3})
        ...
        run.log_metrics(test_metrics, split="test")
        run.log_artifact("model", model_path)
        run.set_best_epoch(12)

On exit the run records duration, status (``completed`` / ``failed``) and any
exception, then writes ``results/experiments/<timestamp>__<modality>__<name>.json``
and appends a row to ``results/experiments/experiment_log.csv``.
"""

from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping

from .config import Config
from .env import describe_environment
from .io_utils import save_json, timestamp, to_serializable
from .logging_utils import get_logger
from .paths import PATHS, ensure_dir

__all__ = ["ExperimentTracker", "load_experiment_log"]

logger = get_logger(__name__)

_INDEX_COLUMNS = [
    "run_id", "timestamp", "modality", "experiment", "status",
    "model", "seed", "batch_size", "learning_rate", "epochs",
    "best_epoch", "duration_sec", "device", "gpu", "git_commit",
    "primary_metric", "primary_value", "metrics_json", "artifact_dir",
]


class ExperimentTracker:
    """Context manager that records everything needed to reproduce a run."""

    def __init__(
        self,
        experiment: str,
        modality: str,
        config: Config | Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        notes: str = "",
        experiments_dir: Path | str | None = None,
        primary_metric: str | None = None,
    ) -> None:
        """
        Args:
            experiment: Short run name, e.g. ``"xgboost_tuned"``.
            modality: ``"clinical"`` | ``"ecg"`` | ``"xray"``.
            config: The full config object; stored verbatim for reproducibility.
            params: Extra hyperparameters to record immediately.
            notes: Free-text note that ends up in the JSON and the CSV.
            experiments_dir: Override the output directory.
            primary_metric: Metric name promoted into the CSV index column.
        """
        self.experiment = experiment
        self.modality = modality
        self.notes = notes
        self.primary_metric = primary_metric

        self.run_id = f"{timestamp()}__{modality}__{experiment}"
        self.dir = ensure_dir(Path(experiments_dir) if experiments_dir else PATHS.experiments)
        self.json_path = self.dir / f"{self.run_id}.json"
        self.index_path = self.dir / "experiment_log.csv"

        config_dict: dict[str, Any] = {}
        if isinstance(config, Config):
            config_dict = config.to_dict()
        elif config is not None:
            config_dict = dict(config)

        self.record: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": experiment,
            "modality": modality,
            "notes": notes,
            "status": "running",
            "started_at": None,
            "finished_at": None,
            "duration_sec": None,
            "seed": config_dict.get("seed"),
            "config": config_dict,
            "params": dict(params or {}),
            "metrics": {},
            "artifacts": {},
            "history": {},
            "best_epoch": None,
            "environment": describe_environment(),
        }
        self._start_time: float | None = None

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "ExperimentTracker":
        self._start_time = time.time()
        self.record["started_at"] = timestamp("%Y-%m-%dT%H:%M:%SZ")
        logger.info("Experiment started: %s", self.run_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        duration = time.time() - (self._start_time or time.time())
        self.record["duration_sec"] = round(duration, 2)
        self.record["finished_at"] = timestamp("%Y-%m-%dT%H:%M:%SZ")

        if exc_type is not None:
            self.record["status"] = "failed"
            self.record["error"] = {
                "type": exc_type.__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb))[-4000:],
            }
            logger.error("Experiment FAILED after %.1fs: %s", duration, exc)
        else:
            self.record["status"] = "completed"
            logger.info("Experiment completed in %.1fs: %s", duration, self.run_id)

        self.save()
        return False  # never swallow the exception

    # -- logging API ----------------------------------------------------------
    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record hyperparameters (merged into any already logged)."""
        self.record["params"].update(to_serializable(dict(params)))

    def log_metrics(self, metrics: Mapping[str, Any], split: str = "test") -> None:
        """Record a block of metrics under a split name (``train``/``val``/``test``)."""
        self.record["metrics"].setdefault(split, {})
        self.record["metrics"][split].update(to_serializable(dict(metrics)))

    def log_history(self, history: Mapping[str, Any]) -> None:
        """Record per-epoch training history for later curve plotting."""
        self.record["history"] = to_serializable(dict(history))

    def log_artifact(self, name: str, path: Path | str) -> None:
        """Record where an output file was written (model, figure, CSV...)."""
        target = Path(path)
        try:
            stored = str(target.relative_to(PATHS.root))
        except ValueError:
            stored = str(target)
        self.record["artifacts"][name] = stored

    def set_best_epoch(self, epoch: int) -> None:
        """Record the epoch whose checkpoint was restored."""
        self.record["best_epoch"] = int(epoch)

    def set_status(self, status: str) -> None:
        self.record["status"] = status

    # -- persistence ----------------------------------------------------------
    def save(self) -> Path:
        """Write the JSON record and append a row to the CSV index."""
        save_json(self.record, self.json_path)
        self._append_index_row()
        return self.json_path

    def _primary(self) -> tuple[str, Any]:
        metrics = self.record["metrics"]
        preferred_split = "test" if "test" in metrics else next(iter(metrics), None)
        if preferred_split is None:
            return ("", "")
        block = metrics[preferred_split]
        if self.primary_metric and self.primary_metric in block:
            return (f"{preferred_split}_{self.primary_metric}", block[self.primary_metric])
        for candidate in ("roc_auc", "macro_roc_auc", "pr_auc", "macro_f1", "f1", "accuracy"):
            if candidate in block:
                return (f"{preferred_split}_{candidate}", block[candidate])
        return ("", "")

    def _append_index_row(self) -> None:
        import json as _json

        params = self.record["params"]
        env = self.record["environment"]
        gpu = env.get("gpu", {})
        metric_name, metric_value = self._primary()

        row = {
            "run_id": self.run_id,
            "timestamp": self.record["started_at"],
            "modality": self.modality,
            "experiment": self.experiment,
            "status": self.record["status"],
            "model": params.get("model", self.record["config"].get("model", {}).get("name", "")),
            "seed": self.record.get("seed"),
            "batch_size": params.get("batch_size", ""),
            "learning_rate": params.get("learning_rate", ""),
            "epochs": params.get("epochs", ""),
            "best_epoch": self.record.get("best_epoch", ""),
            "duration_sec": self.record.get("duration_sec"),
            "device": "cuda" if gpu.get("available") else "cpu",
            "gpu": gpu.get("name", ""),
            "git_commit": env.get("git_commit") or "",
            "primary_metric": metric_name,
            "primary_value": metric_value,
            "metrics_json": _json.dumps(self.record["metrics"].get("test", {}))[:900],
            "artifact_dir": str(self.json_path.parent.name),
        }

        write_header = not self.index_path.exists()
        with self.index_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_INDEX_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def load_experiment_log(path: Path | str | None = None):
    """Load the experiment index as a pandas DataFrame (empty if absent)."""
    import pandas as pd

    target = Path(path) if path else PATHS.experiments / "experiment_log.csv"
    if not target.exists():
        return pd.DataFrame(columns=_INDEX_COLUMNS)
    return pd.read_csv(target)
