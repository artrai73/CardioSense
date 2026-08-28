"""Training utilities shared by the ECG and X-ray pipelines.

The important piece here is :class:`CheckpointManager`. Colab disconnects
routinely kill a GPU session at 40-90 minutes; without resumable checkpoints an
X-ray fine-tune has to restart from epoch 0 every time. Every epoch writes
``last.pt`` (full state: model, optimiser, scheduler, AMP scaler, RNG states,
epoch counter, history) and, when the monitored metric improves, ``best.pt``.
Re-running the same training cell after a disconnect picks up where it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .logging_utils import get_logger
from .paths import ensure_dir

__all__ = ["AverageMeter", "EarlyStopping", "CheckpointManager", "History", "count_parameters"]

logger = get_logger(__name__)


class AverageMeter:
    """Running mean of a scalar (loss, batch time, ...)."""

    def __init__(self, name: str = "meter") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.value = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.sum += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}={self.average:.4f}"


@dataclass
class History:
    """Per-epoch metric store that plugs straight into ``plot_training_curves``."""

    data: dict[str, list[float]] = field(default_factory=dict)

    def append(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.data.setdefault(key, []).append(float(value))

    def last(self, key: str, default: float = float("nan")) -> float:
        values = self.data.get(key)
        return values[-1] if values else default

    def best(self, key: str, mode: str = "max") -> tuple[int, float]:
        """Return ``(1-based epoch, value)`` of the best entry for *key*."""
        values = self.data.get(key, [])
        if not values:
            return (0, float("nan"))
        index = int(max(range(len(values)), key=lambda i: values[i])) if mode == "max" \
            else int(min(range(len(values)), key=lambda i: values[i]))
        return (index + 1, values[index])

    def to_dict(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self.data.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, list[float]] | None) -> "History":
        return cls({k: list(v) for k, v in (data or {}).items()})

    def __len__(self) -> int:
        return max((len(v) for v in self.data.values()), default=0)


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Args:
        patience: Epochs to wait after the last improvement.
        mode: ``"max"`` (higher is better, e.g. AUC) or ``"min"`` (loss).
        min_delta: Minimum change that counts as an improvement.
    """

    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int | None = None) -> bool:
        """Update with the latest metric. Returns ``True`` if it improved."""
        improved = self.is_improvement(value)
        if improved:
            self.best = float(value)
            self.counter = 0
            if epoch is not None:
                self.best_epoch = int(epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping: no improvement in %d epochs (best=%.5f at epoch %d)",
                    self.patience, self.best if self.best is not None else float("nan"),
                    self.best_epoch,
                )
        return improved

    def state_dict(self) -> dict[str, Any]:
        return {
            "best": self.best, "counter": self.counter,
            "should_stop": self.should_stop, "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best = state.get("best")
        self.counter = int(state.get("counter", 0))
        self.should_stop = bool(state.get("should_stop", False))
        self.best_epoch = int(state.get("best_epoch", 0))


class CheckpointManager:
    """Save/restore complete training state so Colab disconnects are survivable.

    Writes two files inside *directory*:

    * ``last.pt`` — rewritten every epoch; used for resume.
    * ``best.pt`` — rewritten whenever the monitored metric improves.

    Example::

        ckpt = CheckpointManager(dir, monitor="val_macro_auc", mode="max")
        start_epoch, history = ckpt.maybe_resume(model, optimizer, scheduler, scaler)
        for epoch in range(start_epoch, epochs):
            ...
            ckpt.save(epoch, model, optimizer, scheduler, scaler,
                      metrics={"val_macro_auc": auc}, history=history.to_dict())
    """

    def __init__(
        self,
        directory: Path | str,
        monitor: str = "val_loss",
        mode: str = "min",
        last_name: str = "last.pt",
        best_name: str = "best.pt",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.directory = ensure_dir(directory)
        self.monitor = monitor
        self.mode = mode
        self.last_path = self.directory / last_name
        self.best_path = self.directory / best_name
        self.best_value: float | None = None
        self.best_epoch: int = 0

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _rng_state() -> dict[str, Any]:
        import random

        import numpy as np
        import torch

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        import random

        import numpy as np
        import torch

        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu")
                                else state["torch"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except (KeyError, RuntimeError, TypeError) as exc:  # pragma: no cover
            logger.warning("Could not restore RNG state (%s); continuing with a fresh one.", exc)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == "max" else value < self.best_value

    # -- public API -----------------------------------------------------------
    def save(
        self,
        epoch: int,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        metrics: Mapping[str, float] | None = None,
        history: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Persist state. Returns the paths written.

        The improvement check happens **before** anything is written, so both
        files record the up-to-date best value. Writing ``last.pt`` first and
        updating afterwards would leave it holding the *previous* epoch's best —
        which then gets restored on resume, so early stopping and ``_is_better``
        would both restart from a stale, worse baseline and could overwrite
        ``best.pt`` with an inferior checkpoint.
        """
        import torch

        metrics = dict(metrics or {})

        current = metrics.get(self.monitor)
        improved = current is not None and self._is_better(float(current))
        if improved:
            self.best_value = float(current)
            self.best_epoch = int(epoch)

        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "history": dict(history or {}),
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "rng_state": self._rng_state(),
            "extra": dict(extra or {}),
        }

        written = {"last": self.last_path}
        torch.save(payload, self.last_path)

        if improved:
            torch.save(payload, self.best_path)
            written["best"] = self.best_path
            logger.info("New best %s = %.5f at epoch %d -> %s",
                        self.monitor, self.best_value, epoch, self.best_path.name)
        return written

    def maybe_resume(
        self,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: Any = None,
        enabled: bool = True,
    ) -> tuple[int, dict[str, list[float]]]:
        """Restore from ``last.pt`` if it exists.

        Returns:
            ``(start_epoch, history_dict)``. ``(0, {})`` when starting fresh.
        """
        if not enabled or not self.last_path.exists():
            return 0, {}

        import torch

        # weights_only=False: our payload contains RNG state and plain Python
        # objects, not just tensors. The file is written by this same codebase.
        try:
            payload = torch.load(self.last_path, map_location=map_location, weights_only=False)
        except TypeError:  # torch < 2.4 has no weights_only kwarg
            payload = torch.load(self.last_path, map_location=map_location)

        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])

        self.best_value = payload.get("best_value")
        self.best_epoch = int(payload.get("best_epoch", 0))
        self._restore_rng(payload.get("rng_state"))

        start_epoch = int(payload["epoch"]) + 1
        logger.info("Resumed from %s at epoch %d (best %s = %s)",
                    self.last_path.name, start_epoch, self.monitor, self.best_value)
        return start_epoch, dict(payload.get("history", {}))

    def load_best(self, model: Any, map_location: Any = None) -> dict[str, Any]:
        """Load ``best.pt`` weights into *model*; returns the checkpoint payload."""
        import torch

        path = self.best_path if self.best_path.exists() else self.last_path
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found in {self.directory}")
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model_state"])
        logger.info("Loaded weights from %s (epoch %s)", path.name, payload.get("epoch"))
        return payload


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """Number of parameters, reported in the model-comparison tables."""
    params = model.parameters()
    return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))
