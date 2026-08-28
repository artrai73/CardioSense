"""Training loop for the chest X-ray model.

Two things distinguish this from the ECG trainer:

**Two-stage fine-tuning inside one optimiser.** The optimiser is created once with
two parameter groups — head and backbone — at different learning rates. Stage 1
freezes the backbone (``requires_grad = False``, so its group receives no
updates); at ``freeze_backbone_epochs`` the later blocks are unfrozen and start
training at the smaller rate. Doing it this way rather than rebuilding the
optimiser keeps checkpoint resume simple and correct: there is one optimiser state
to save and restore, with no stage-dependent shape changes.

**Monitored metric is PR-AUC.** At ~2.5% prevalence, ROC-AUC is dominated by the
enormous negative class and can look respectable while precision is useless, and
accuracy is worse than useless. PR-AUC responds to what actually matters here: of
the images flagged, how many were right.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..common.compat import autocast_context, make_grad_scaler
from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.metrics import binary_metrics
from ..common.paths import ensure_dir
from ..common.training import AverageMeter, CheckpointManager, EarlyStopping, History
from .models import XrayDenseNet121, set_backbone_trainable

__all__ = ["build_optimizer", "build_scheduler", "build_criterion", "train_model"]

logger = get_logger("xray.trainer")


def build_optimizer(model: XrayDenseNet121, cfg: Config) -> torch.optim.Optimizer:
    """Build one optimiser with separate head and backbone parameter groups.

    Group 0 (head) uses ``training.learning_rate``; group 1 (backbone) uses
    ``training.finetune_learning_rate``, which is much smaller. Pretrained
    features need only gentle adjustment; a head-sized learning rate applied to
    them erases what transfer learning provided.
    """
    head_lr = float(cfg.training.learning_rate)
    backbone_lr = float(cfg.training.get("finetune_learning_rate", head_lr / 10))
    weight_decay = float(cfg.training.get("weight_decay", 0.0))

    groups = [
        {"params": list(model.classifier.parameters()), "lr": head_lr, "name": "head"},
        {"params": list(model.features.parameters()), "lr": backbone_lr, "name": "backbone"},
    ]

    name = str(cfg.training.get("optimizer", "adamw")).lower()
    if name == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = torch.optim.Adam(groups, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = torch.optim.SGD(groups, momentum=0.9, weight_decay=weight_decay,
                                    nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {name!r}.")

    logger.info("Optimizer %s: head lr %.2e, backbone lr %.2e", name, head_lr, backbone_lr)
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> Any:
    """Cosine schedule with linear warmup, applied to both parameter groups.

    Warmup matters here specifically: the classification head is randomly
    initialised, so the first few batches produce large gradients. Ramping the
    learning rate from zero over the first epoch stops those from destabilising
    training before the head has found a sensible region.

    Implemented as a ``LambdaLR`` multiplier so each group keeps its own base rate.
    """
    scheduler_cfg = cfg.training.get("scheduler", {}) or {}
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    if name in {"none", "null"}:
        return None

    epochs = int(cfg.training.epochs)
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_factor = float(scheduler_cfg.get("min_lr", 1e-6)) / max(
        float(cfg.training.learning_rate), 1e-12
    )

    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 2)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )

    if name != "cosine":
        raise ValueError(f"Unknown scheduler {name!r}; use cosine or reduce_on_plateau.")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(cosine, min_factor)

    logger.info("Cosine schedule over %d epochs with %d warmup epoch(s).",
                epochs, warmup_epochs)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_criterion(cfg: Config, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """``BCEWithLogitsLoss``, optionally with positive-class weighting.

    ``pos_weight`` is the single imbalance correction used here. See
    ``class_imbalance`` in the config for why it is not combined with a weighted
    sampler.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    grad_clip: float | None,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """One training pass."""
    from tqdm.auto import tqdm

    model.train()
    loss_meter = AverageMeter("train_loss")
    start = time.time()

    progress = tqdm(loader, desc=f"epoch {epoch + 1}/{total_epochs} [train]", leave=False)
    for batch in progress:
        images, targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp, device.type):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    return {"train_loss": loss_meter.average, "train_seconds": round(time.time() - start, 2)}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    """One validation pass, reporting loss plus threshold-free metrics.

    Metrics at a fixed 0.5 threshold are computed only so F1 is comparable across
    epochs. The reported operating threshold is tuned once, after training.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    probabilities: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    for batch in loader:
        images, targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        with autocast_context(use_amp, device.type):
            logits = model(images)
            loss = criterion(logits, targets_device)
        loss_meter.update(loss.item(), images.size(0))
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        targets_all.append(targets.numpy().ravel())

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(targets_all).astype(int)
    metrics = binary_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)

    return {
        "val_loss": loss_meter.average,
        "val_roc_auc": float(metrics.get("roc_auc", float("nan"))),
        "val_pr_auc": float(metrics.get("pr_auc", float("nan"))),
        "val_f1": float(metrics["f1"]),
        "val_recall": float(metrics["recall"]),
        "val_precision": float(metrics["precision"]),
    }


def train_model(
    model: XrayDenseNet121,
    loaders: dict[str, Any],
    cfg: Config,
    device: torch.device,
    checkpoint_dir: Path | str,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Train with staged unfreezing, early stopping and resumable checkpointing.

    Returns:
        History, best epoch, monitored metric, timing and the unfreeze report.
    """
    training = cfg.training
    epochs = int(training.epochs)
    monitor = str(training.early_stopping.get("monitor", "val_pr_auc"))
    mode = str(training.early_stopping.get("mode", "max"))
    freeze_epochs = int(cfg.model.get("freeze_backbone_epochs", 0))
    unfreeze_from = cfg.model.get("unfreeze_from_block")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_criterion(cfg, pos_weight.to(device) if pos_weight is not None else None)

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
        patience=int(training.early_stopping.get("patience", 4)),
        mode=mode,
        min_delta=float(training.early_stopping.get("min_delta", 0.0)),
    )
    if checkpoints.best_value is not None:
        stopper.best = checkpoints.best_value
        stopper.best_epoch = checkpoints.best_epoch

    unfreeze_report: dict[str, Any] = {}

    # Resuming past the unfreeze point must restore the unfrozen state, or the
    # backbone would silently stay frozen for the rest of the run.
    if start_epoch >= freeze_epochs and freeze_epochs > 0:
        unfreeze_report = set_backbone_trainable(model, True, from_block=unfreeze_from)
        logger.info("Resumed past the unfreeze epoch; backbone state restored.")

    if start_epoch >= epochs:
        logger.info("Checkpoint already at epoch %d of %d; nothing to train.",
                    start_epoch, epochs)
        checkpoints.load_best(model, map_location=device)
        return {"history": history.to_dict(), "best_epoch": int(checkpoints.best_epoch) + 1,
                "best_value": checkpoints.best_value, "monitor": monitor,
                "epochs_run": 0, "resumed_from": start_epoch, "total_seconds": 0.0,
                "unfreeze": unfreeze_report}

    logger.info("Training DenseNet121: epochs %d-%d, device=%s, AMP=%s, monitor=%s",
                start_epoch + 1, epochs, device, use_amp, monitor)

    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        # -- stage transition ------------------------------------------------
        if freeze_epochs > 0 and epoch == freeze_epochs:
            logger.info("Stage 2: unfreezing the backbone from %s at lr %.2e",
                        unfreeze_from, optimizer.param_groups[1]["lr"])
            unfreeze_report = set_backbone_trainable(model, True, from_block=unfreeze_from)

        stage = "head-only" if (freeze_epochs > 0 and epoch < freeze_epochs) else "fine-tune"

        train_stats = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler,
            use_amp, grad_clip, epoch, epochs,
        )
        val_stats = validate_one_epoch(model, loaders["val"], criterion, device, use_amp)

        head_lr = optimizer.param_groups[0]["lr"]
        history.append(
            train_loss=train_stats["train_loss"],
            val_loss=val_stats["val_loss"],
            val_roc_auc=val_stats["val_roc_auc"],
            val_pr_auc=val_stats["val_pr_auc"],
            val_f1=val_stats["val_f1"],
            val_recall=val_stats["val_recall"],
            learning_rate=head_lr,
        )

        monitored = val_stats.get(monitor, val_stats["val_loss"])
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
            extra={"stage": stage, "unfrozen": bool(unfreeze_report)},
        )

        logger.info(
            "epoch %2d/%d [%s] | train loss %.4f | val loss %.4f | PR-AUC %.4f | "
            "ROC-AUC %.4f | recall %.3f | lr %.2e%s",
            epoch + 1, epochs, stage, train_stats["train_loss"], val_stats["val_loss"],
            val_stats["val_pr_auc"], val_stats["val_roc_auc"], val_stats["val_recall"],
            head_lr, "  <- best" if improved else "",
        )

        if stopper.should_stop:
            logger.info("Early stopping at epoch %d (best %s = %.4f at epoch %d).",
                        epoch + 1, monitor, stopper.best or float("nan"), stopper.best_epoch)
            break

    total_seconds = time.time() - run_start
    checkpoints.load_best(model, map_location=device)
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
        "freeze_backbone_epochs": freeze_epochs,
        "unfreeze_from_block": unfreeze_from,
        "unfreeze": unfreeze_report,
    }
    logger.info("Training complete in %.1fs. Best %s = %.4f at epoch %d.",
                total_seconds, monitor, checkpoints.best_value or float("nan"),
                best_epoch_display)
    return result
