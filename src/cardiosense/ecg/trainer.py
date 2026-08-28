"""Training loop for the ECG models.

Built around surviving Colab. Every epoch writes ``last.pt`` containing model,
optimiser, scheduler, AMP scaler, RNG state, epoch counter and full history; a
better epoch also writes ``best.pt``. Re-running the same cell after a dropped
runtime resumes at the next epoch instead of starting over.

Monitoring choice: ``val_macro_auc``, not validation loss. Loss is dominated by
the common classes, so it can improve while performance on HYP — the rare and
clinically interesting one — degrades. Macro-averaged AUC weights all five
classes equally and is threshold-free, so it does not conflate "the model got
better" with "the operating point drifted".
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..common.compat import autocast_context, make_grad_scaler
from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.metrics import multilabel_metrics
from ..common.paths import ensure_dir
from ..common.training import AverageMeter, CheckpointManager, EarlyStopping, History

__all__ = ["build_optimizer", "build_scheduler", "build_criterion", "train_model",
           "train_one_epoch", "validate_one_epoch"]

logger = get_logger("ecg.trainer")


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build the optimiser named by ``training.optimizer``.

    AdamW rather than Adam: it decouples weight decay from the adaptive learning
    rate, so the decay actually regularises instead of being scaled away by the
    per-parameter step size.
    """
    name = str(cfg.training.get("optimizer", "adamw")).lower()
    lr = float(cfg.training.learning_rate)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer {name!r}; use adamw, adam or sgd.")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int = 0) -> Any:
    """Build the LR scheduler named by ``training.scheduler.name``.

    ``reduce_on_plateau`` is the default: it responds to the monitored metric
    rather than to a fixed schedule, which suits a run whose length is decided by
    early stopping rather than known in advance.
    """
    scheduler_cfg = cfg.training.get("scheduler", {}) or {}
    name = str(scheduler_cfg.get("name", "reduce_on_plateau")).lower()

    if name in {"none", "null"}:
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 3)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.training.epochs),
            eta_min=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler {name!r}.")


def build_criterion(cfg: Config, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Build the loss function.

    ``BCEWithLogitsLoss`` is the correct loss for multi-label classification: it
    treats each class as an independent binary decision, which is exactly the
    task. Cross-entropy over a softmax would force the five class scores to
    compete for a fixed probability budget and make co-occurring labels
    impossible to express.
    """
    loss_name = str(cfg.training.get("loss", "bce_with_logits")).lower()
    if loss_name != "bce_with_logits":
        raise ValueError(
            f"Loss {loss_name!r} is not appropriate for a multi-label task. "
            "Use bce_with_logits."
        )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    grad_clip: float | None = None,
    epoch: int = 0,
    total_epochs: int = 0,
) -> dict[str, float]:
    """One training pass. Returns the mean loss and timing."""
    from tqdm.auto import tqdm

    model.train()
    loss_meter = AverageMeter("train_loss")
    start = time.time()

    progress = tqdm(loader, desc=f"epoch {epoch + 1}/{total_epochs} [train]", leave=False)
    for waveforms, labels in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp, device.type):
            logits = model(waveforms)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip:
            # Unscale before clipping, or the clip threshold applies to scaled
            # gradients and effectively does nothing.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), waveforms.size(0))
        progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    return {"train_loss": loss_meter.average, "train_seconds": round(time.time() - start, 2)}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    use_amp: bool = False,
) -> dict[str, Any]:
    """One validation pass. Returns loss plus threshold-free multi-label metrics.

    Metrics here use a fixed 0.5 threshold purely so that F1 is comparable across
    epochs. The reported operating thresholds are tuned separately, once, after
    training — tuning them every epoch would make early stopping chase the
    threshold search rather than the model.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with autocast_context(use_amp, device.type):
            logits = model(waveforms)
            loss = criterion(logits, labels_device)
        loss_meter.update(loss.item(), waveforms.size(0))
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels_all.append(labels.numpy())

    y_prob = np.vstack(probabilities)
    y_true = np.vstack(labels_all)
    metrics = multilabel_metrics(y_true, y_prob, thresholds=0.5, class_names=class_names)

    return {
        "val_loss": loss_meter.average,
        "val_macro_auc": float(metrics["macro_roc_auc"]),
        "val_macro_pr_auc": float(metrics["macro_pr_auc"]),
        "val_macro_f1": float(metrics["macro_f1"]),
        "per_class": metrics["per_class"],
    }


def train_model(
    model: nn.Module,
    loaders: dict[str, Any],
    cfg: Config,
    device: torch.device,
    checkpoint_dir: Path | str,
    pos_weight: torch.Tensor | None = None,
    experiment_name: str = "ecg",
) -> dict[str, Any]:
    """Train with early stopping, LR scheduling and resumable checkpointing.

    Args:
        model: The model, already moved to ``device``.
        loaders: ``{"train": ..., "val": ...}``.
        cfg: ECG configuration.
        device: Target device.
        checkpoint_dir: Where ``last.pt`` and ``best.pt`` are written.
        pos_weight: Per-class positive weights for the loss.
        experiment_name: Used in log lines.

    Returns:
        Dict with history, best epoch, best metric value and timing.
    """
    training = cfg.training
    epochs = int(training.epochs)
    class_names = list(cfg.task.classes)

    monitor = str(training.early_stopping.get("monitor", "val_macro_auc"))
    mode = str(training.early_stopping.get("mode", "max"))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_criterion(
        cfg, pos_weight.to(device) if pos_weight is not None else None
    )

    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(enabled=use_amp, device_type=device.type)
    grad_clip = training.get("grad_clip_norm")

    checkpoints = CheckpointManager(ensure_dir(checkpoint_dir), monitor=monitor, mode=mode)
    start_epoch, history_data = checkpoints.maybe_resume(
        model, optimizer, scheduler, scaler,
        map_location=device, enabled=bool(training.get("resume", True)),
    )
    history = History.from_dict(history_data)

    stopper = EarlyStopping(
        patience=int(training.early_stopping.get("patience", 8)),
        mode=mode,
        min_delta=float(training.early_stopping.get("min_delta", 0.0)),
    )
    if checkpoints.best_value is not None:
        stopper.best = checkpoints.best_value
        stopper.best_epoch = checkpoints.best_epoch

    if start_epoch >= epochs:
        logger.info("Checkpoint is already at epoch %d of %d; nothing left to train.",
                    start_epoch, epochs)
        checkpoints.load_best(model, map_location=device)
        return {"history": history.to_dict(), "best_epoch": int(checkpoints.best_epoch) + 1,
                "best_value": checkpoints.best_value, "monitor": monitor,
                "epochs_run": 0, "resumed_from": start_epoch, "total_seconds": 0.0}

    logger.info("Training %s: epochs %d-%d, device=%s, AMP=%s, monitor=%s (%s)",
                experiment_name, start_epoch + 1, epochs, device, use_amp, monitor, mode)

    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler,
            use_amp, grad_clip, epoch, epochs,
        )
        val_stats = validate_one_epoch(
            model, loaders["val"], criterion, device, class_names, use_amp
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            train_loss=train_stats["train_loss"],
            val_loss=val_stats["val_loss"],
            val_macro_auc=val_stats["val_macro_auc"],
            val_macro_pr_auc=val_stats["val_macro_pr_auc"],
            val_macro_f1=val_stats["val_macro_f1"],
            learning_rate=current_lr,
        )

        monitored = val_stats[monitor] if monitor in val_stats else val_stats["val_loss"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitored)
            else:
                scheduler.step()

        improved = stopper.step(monitored, epoch=epoch + 1)
        checkpoints.save(
            epoch, model, optimizer, scheduler, scaler,
            metrics={monitor: monitored, "val_loss": val_stats["val_loss"]},
            history=history.to_dict(),
            extra={"class_names": class_names},
        )

        logger.info(
            "epoch %2d/%d | train loss %.4f | val loss %.4f | macro AUC %.4f | "
            "macro F1 %.4f | lr %.2e%s",
            epoch + 1, epochs, train_stats["train_loss"], val_stats["val_loss"],
            val_stats["val_macro_auc"], val_stats["val_macro_f1"], current_lr,
            "  <- best" if improved else "",
        )

        if stopper.should_stop:
            logger.info("Early stopping at epoch %d (best %s = %.4f at epoch %d).",
                        epoch + 1, monitor, stopper.best or float("nan"), stopper.best_epoch)
            break

    total_seconds = time.time() - run_start
    checkpoints.load_best(model, map_location=device)

    # CheckpointManager stores the 0-based loop index (which is what resume needs);
    # everything user-facing counts epochs from 1.
    best_epoch_display = int(checkpoints.best_epoch) + 1

    result = {
        "history": history.to_dict(),
        "monitor": monitor,
        "best_epoch": best_epoch_display,
        "best_value": checkpoints.best_value,
        "epochs_run": len(history) - start_epoch,
        "resumed_from": start_epoch,
        "total_seconds": round(total_seconds, 2),
        "seconds_per_epoch": round(total_seconds / max(len(history) - start_epoch, 1), 2),
        "device": str(device),
        "amp": use_amp,
    }
    logger.info("Training complete in %.1fs. Best %s = %.4f at epoch %d.",
                total_seconds, monitor, checkpoints.best_value or float("nan"),
                best_epoch_display)
    return result
