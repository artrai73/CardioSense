"""End-to-end training entry point for the chest X-ray pipeline.

Run it::

    python -m cardiosense.xray.train
    python -m cardiosense.xray.train --set training.epochs=2 --skip-baseline
    python -m cardiosense.xray.train --set dataset.max_images=2000

Pipeline order::

    verify ChestX-ray14 -> metadata -> binary target + view filter
    -> PATIENT-LEVEL split -> transforms (augment train only)
    -> majority + pixel baselines (X-A) -> DenseNet121 two-stage fine-tune (X-B)
    -> threshold on val -> evaluate test ONCE -> error analysis -> Grad-CAM -> save

Training resumes automatically from ``models/xray/checkpoints/last.pt`` after a
Colab disconnect.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common.config import Config, load_config
from ..common.env import get_device, print_environment
from ..common.experiment import ExperimentTracker
from ..common.io_utils import save_dataframe, save_json, save_pickle
from ..common.logging_utils import get_logger, log_section
from ..common.paths import PATHS, ensure_dir
from ..common.plots import plot_training_curves
from ..common.seeding import set_seed
from ..common.training import count_parameters
from . import evaluate as ev
from .baseline import build_feature_matrix, majority_class_baseline, train_pixel_baseline
from .data import (
    build_target,
    describe_dataset,
    load_metadata,
    resolve_nih_root,
    split_by_patient,
    verify_dataset,
)
from .dataset import ChestXrayDataset, build_dataloaders, compute_pos_weight
from .explain import run_gradcam_analysis
from .models import build_model, model_summary
from .preprocessing import build_transforms
from .trainer import train_model

__all__ = ["run_pipeline", "main"]

logger = get_logger("xray.train")


def _parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value`` into a typed pair."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    key, raw = text.split("=", 1)
    lowered = raw.strip().lower()
    value: Any
    if lowered in {"true", "false"}:
        value = lowered == "true"
    elif lowered in {"none", "null"}:
        value = None
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
    return key.strip(), value


def _resolve_out(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PATHS.root / path


def run_pipeline(
    cfg: Config,
    skip_baseline: bool = False,
    skip_explain: bool = False,
) -> dict[str, Any]:
    """Execute the full X-ray pipeline."""
    started = time.time()
    set_seed(int(cfg.seed), strict=bool(cfg.get("strict_determinism", False)))
    device = get_device()

    results_dir = ensure_dir(_resolve_out(cfg.output.results_dir))
    models_dir = ensure_dir(_resolve_out(cfg.output.models_dir))
    checkpoint_dir = ensure_dir(models_dir / "checkpoints")
    summary: dict[str, Any] = {"config_path": cfg.get("_config_path")}

    # ------------------------------------------------------------- 1. dataset
    log_section(logger, "1. dataset verification")
    root = resolve_nih_root(cfg)
    summary["dataset_check"] = verify_dataset(cfg, root)
    images_dir = root / str(cfg.dataset.images_dir)
    frame = load_metadata(cfg, root)

    # -------------------------------------------------------------- 2. target
    log_section(logger, "2. target extraction and filtering")
    frame, target_report = build_target(frame, cfg)
    save_json(target_report, results_dir / "target_report.json")
    summary["target"] = target_report
    summary["dataset_description"] = describe_dataset(frame, cfg)
    save_json(summary["dataset_description"], results_dir / "dataset_description.json")

    # --------------------------------------------------------------- 3. split
    log_section(logger, "3. PATIENT-LEVEL split")
    splits = split_by_patient(frame, cfg, root)
    save_json(splits["summary"], results_dir / "split_summary.json")
    summary["split"] = splits["summary"]

    y_train = splits["train"].target.to_numpy()
    y_val = splits["val"].target.to_numpy()
    y_test = splits["test"].target.to_numpy()
    experiments: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ 4. baselines
    if not skip_baseline:
        log_section(logger, "4. baselines (Experiment X-A)")

        majority = majority_class_baseline(y_train, y_test)
        experiments["majority_class"] = {
            "metrics": {"at_tuned_threshold": majority["metrics"]},
            "parameters": 1, "train_seconds": 0.0,
        }

        start = time.time()
        X_train = build_feature_matrix(splits["train"], images_dir, cfg)
        X_val = build_feature_matrix(splits["val"], images_dir, cfg)
        X_test = build_feature_matrix(splits["test"], images_dir, cfg)

        pixel = train_pixel_baseline(X_train, y_train, X_val, cfg, X_test=X_test)
        threshold_pixel, _info = ev.tune_threshold(y_val, pixel["val_prob"], cfg)
        metrics_pixel = ev.evaluate_binary(y_test, pixel["test_prob"], threshold_pixel,
                                           cfg, split_name="test/pixel_baseline")
        experiments["logreg_pixel_features"] = {
            "metrics": metrics_pixel,
            "parameters": pixel["parameters"],
            "train_seconds": round(time.time() - start, 2),
        }
        save_pickle(pixel["model"], models_dir / "xray_baseline.pkl")
        save_json({"majority": majority["metrics"], "pixel": metrics_pixel},
                  results_dir / "baseline_metrics.json")
    else:
        logger.info("Skipping baselines (--skip-baseline).")

    # ------------------------------------------------------------ 5. DenseNet
    log_section(logger, "5. DenseNet121 transfer learning (Experiment X-B)")
    transforms = build_transforms(cfg)
    loaders = build_dataloaders(splits, images_dir, transforms, cfg)

    pos_weight = None
    method = str(cfg.class_imbalance.get("method", "pos_weight")).lower()
    if method == "pos_weight":
        pos_weight = compute_pos_weight(y_train)
    elif method == "weighted_sampler":
        logger.info("Using WeightedRandomSampler; pos_weight is NOT applied as well "
                    "(stacking both double-counts the correction).")

    model = build_model(cfg).to(device)
    architecture = model_summary(
        model, input_shape=(2, 3, int(cfg.preprocessing.image_size),
                            int(cfg.preprocessing.image_size))
    )
    logger.info("Architecture: %s", architecture)
    summary["architecture"] = architecture

    training_result = train_model(model, loaders, cfg, device, checkpoint_dir,
                                  pos_weight=pos_weight)
    summary["training"] = training_result

    plot_training_curves(
        training_result["history"], results_dir / "training_curve.png",
        loss_keys=("train_loss", "val_loss"),
        metric_keys=("val_pr_auc", "val_roc_auc", "val_recall"),
        best_epoch=training_result["best_epoch"],
        title="DenseNet121 — training history",
    )

    # -------------------------------------------------- 6. threshold + testing
    log_section(logger, "6. threshold (validation) and test evaluation")
    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"

    val_prob, val_true, _val_idx = ev.predict_probabilities(model, loaders["val"], device, use_amp)
    threshold, threshold_info = ev.tune_threshold(val_true, val_prob, cfg)
    save_json(threshold_info, results_dir / "threshold.json")

    test_prob, test_true, test_idx = ev.predict_probabilities(
        model, loaders["test"], device, use_amp
    )
    # The test loader is not shuffled, but reorder defensively so predictions
    # always align with the split frame regardless of loader configuration.
    order = np.argsort(test_idx)
    test_prob, test_true = test_prob[order], test_true[order]

    test_metrics = ev.evaluate_binary(test_true, test_prob, threshold, cfg, split_name="test")
    experiments["densenet121"] = {
        "metrics": test_metrics,
        "parameters": count_parameters(model, trainable_only=False),
        "train_seconds": training_result["total_seconds"],
    }
    summary["test_metrics"] = test_metrics
    summary["threshold"] = threshold_info

    extra_curves = {}
    if "logreg_pixel_features" in experiments and not skip_baseline:
        extra_curves["Pixel-feature LogReg"] = (y_test, pixel["test_prob"])
    ev.plot_evaluation_figures(test_true, test_prob, threshold, results_dir,
                               model_name="DenseNet121", extra_curves=extra_curves)

    comparison = ev.build_comparison_table(experiments)
    save_dataframe(comparison, results_dir / "model_comparison.csv")
    logger.info("\n%s", comparison.to_string(index=False))
    summary["model_comparison"] = comparison.to_dict(orient="records")

    # Did DenseNet121 actually beat the baselines? Spec 6.3 asks the baselines to
    # establish that the deep model provides real predictive capability.
    recommendation = ev.recommend_model(experiments)
    save_json(recommendation, results_dir / "model_recommendation.json")
    summary["recommendation"] = recommendation

    # ---------------------------------------------------------- 7. error analysis
    log_section(logger, "7. error analysis")
    errors = ev.export_errors(splits["test"], test_true, test_prob, threshold, cfg,
                              results_dir / "errors", prefix="test")
    summary["errors"] = errors

    # --------------------------------------------------------------- 8. Grad-CAM
    if not skip_explain:
        log_section(logger, "8. explainability (Grad-CAM)")
        try:
            n_per_category = max(1, int(cfg.explainability.get("n_examples", 12)) // 4)
            selection = ev.select_error_examples(test_true, test_prob, threshold,
                                                 n_per_category=n_per_category)
            gradcam_dataset = ChestXrayDataset(
                splits["test"], images_dir, transforms["test"],
                image_column=str(cfg.dataset.image_column), return_index=False,
            )
            summary["explanations"] = run_gradcam_analysis(
                model, splits["test"].reset_index(drop=True), gradcam_dataset, selection,
                test_true, test_prob, threshold, cfg, device, results_dir / "gradcam",
            )
        except Exception as exc:  # noqa: BLE001 - XAI must not lose a trained model
            logger.warning("Grad-CAM failed (%s); the model is still saved.", exc)
    else:
        logger.info("Skipping Grad-CAM (--skip-explain).")

    # -------------------------------------------------------------- 9. artifacts
    log_section(logger, "9. saving artifacts")
    model_path = models_dir / cfg.output.model_file
    torch.save({
        "model_state": model.state_dict(),
        "model_name": "densenet121",
        "threshold": float(threshold),
    }, model_path)

    inference_config = {
        "model_version": str(cfg.output.model_version),
        "model_name": "densenet121",
        "n_classes": int(cfg.model.get("n_classes", 1)),
        "dropout": float(cfg.model.get("dropout", 0.2)),
        "target_label": str(cfg.dataset.target_label),
        "label_mapping": {"0": f"no {cfg.dataset.target_label}".lower(),
                          "1": str(cfg.dataset.target_label).lower()},
        "threshold": round(float(threshold), 4),
        "image_size": int(cfg.preprocessing.image_size),
        "normalize_mean": list(cfg.preprocessing.normalize_mean),
        "normalize_std": list(cfg.preprocessing.normalize_std),
        "to_rgb": bool(cfg.preprocessing.get("to_rgb", True)),
        "view_filter": cfg.dataset.get("filter_view"),
        "training_prevalence": round(float(y_train.mean()), 5),
        "negative_subsampling_ratio": cfg.dataset.get("negative_ratio"),
        "prevalence_warning": (
            "Probabilities are conditioned on the sampled training prevalence, which "
            "differs from the population prevalence because negatives were subsampled. "
            "Correct for the prior before reading these as population risk."
        ),
    }
    config_path = save_json(inference_config, models_dir / cfg.output.config_file)

    metadata_path = save_json({
        **inference_config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modality": "xray",
        "dataset": {"name": str(cfg.dataset.name), "root": str(root)},
        "architecture": architecture,
        "split_summary": splits["summary"],
        "target_report": target_report,
        "training": training_result,
        "test_metrics": test_metrics,
        "threshold_info": threshold_info,
        "class_imbalance": cfg.class_imbalance.to_dict(),
        "augmentation": cfg.augmentation.to_dict(),
        "training_config": cfg.to_dict(),
    }, models_dir / cfg.output.metadata_file)

    save_json({
        "model": "densenet121",
        "test": test_metrics,
        "model_comparison": comparison.to_dict(orient="records"),
        "recommendation": recommendation,
        "split": splits["summary"],
        "threshold": threshold_info,
        "training": training_result,
    }, results_dir / "metrics.json")

    summary["artifacts"] = {
        "model": str(model_path), "config": str(config_path),
        "metadata": str(metadata_path), "checkpoints": str(checkpoint_dir),
    }
    summary["duration_sec"] = round(time.time() - started, 2)

    log_section(logger, "xray pipeline complete")
    tuned = test_metrics["at_tuned_threshold"]
    logger.info("PR-AUC %.4f (chance %.4f) | ROC-AUC %.4f | recall %.3f | best epoch %s",
                tuned.get("pr_auc", float("nan")), test_metrics["prevalence"],
                tuned.get("roc_auc", float("nan")), tuned["recall"],
                training_result["best_epoch"])
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the CardioSense X-ray pipeline.")
    parser.add_argument("--config", default="xray", help="Config name or YAML path.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Override a config value.")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--experiment-name", default="xray_pipeline")
    args = parser.parse_args(argv)

    overrides = dict(_parse_override(item) for item in args.overrides)
    cfg = load_config(args.config, overrides=overrides)
    print_environment()

    with ExperimentTracker(args.experiment_name, modality="xray", config=cfg,
                           primary_metric="pr_auc") as run:
        summary = run_pipeline(cfg, skip_baseline=args.skip_baseline,
                               skip_explain=args.skip_explain)
        run.log_params({
            "model": "densenet121",
            "pretrained": bool(cfg.model.get("pretrained", True)),
            "batch_size": int(cfg.training.batch_size),
            "learning_rate": float(cfg.training.learning_rate),
            "epochs": int(cfg.training.epochs),
            "parameters": summary["architecture"]["total"],
            "class_imbalance": str(cfg.class_imbalance.get("method")),
            "n_train": summary["split"]["splits"]["train"]["n_images"],
        })
        run.log_metrics(
            {k: v for k, v in summary["test_metrics"]["at_tuned_threshold"].items()
             if isinstance(v, (int, float))},
            split="test",
        )
        run.log_history(summary["training"]["history"])
        run.set_best_epoch(int(summary["training"]["best_epoch"]))
        for name, path in summary["artifacts"].items():
            run.log_artifact(name, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
