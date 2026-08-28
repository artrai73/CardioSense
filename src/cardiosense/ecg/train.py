"""End-to-end training entry point for the ECG pipeline.

Run it::

    python -m cardiosense.ecg.train
    python -m cardiosense.ecg.train --set training.epochs=2 --skip-baseline
    python -m cardiosense.ecg.train --set model.name=resnet1d --experiment-name resnet_run

Pipeline order::

    verify PTB-XL -> metadata -> superclass labels -> official fold split
    -> waveform cache (preprocess once) -> statistical baseline (E-A)
    -> 1D CNN or ResNet-1D (E-B / E-C) -> per-class thresholds on val
    -> evaluate test ONCE -> error analysis -> Integrated Gradients -> save

Training resumes automatically from ``models/ecg/checkpoints/last.pt`` after a
Colab disconnect, so re-running this command after a dropped runtime continues
rather than restarting.
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
from .baseline import extract_feature_matrix, feature_names, train_baseline
from .data import (
    build_superclass_labels,
    describe_dataset,
    load_metadata,
    resolve_ptbxl_root,
    split_by_fold,
    verify_dataset,
)
from .dataset import build_dataloaders, compute_pos_weight
from .explain import run_ig_analysis
from .models import build_model, model_summary
from .preprocessing import build_waveform_cache
from .trainer import train_model

__all__ = ["run_pipeline", "main"]

logger = get_logger("ecg.train")


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
    force_cache: bool = False,
) -> dict[str, Any]:
    """Execute the full ECG pipeline.

    Args:
        cfg: ECG configuration.
        skip_baseline: Skip the classical baseline (Experiment E-A).
        skip_explain: Skip Integrated Gradients.
        force_cache: Rebuild the waveform cache even if it exists.

    Returns:
        Summary of every stage.
    """
    started = time.time()
    set_seed(int(cfg.seed), strict=bool(cfg.get("strict_determinism", False)))
    device = get_device()

    results_dir = ensure_dir(_resolve_out(cfg.output.results_dir))
    models_dir = ensure_dir(_resolve_out(cfg.output.models_dir))
    checkpoint_dir = ensure_dir(models_dir / "checkpoints")
    classes = list(cfg.task.classes)
    summary: dict[str, Any] = {"config_path": cfg.get("_config_path")}

    # ------------------------------------------------------------ 1. dataset
    log_section(logger, "1. dataset verification")
    root = resolve_ptbxl_root(cfg)
    summary["dataset_check"] = verify_dataset(cfg, root)
    database, statements = load_metadata(cfg, root)

    # ------------------------------------------------------------- 2. labels
    log_section(logger, "2. label construction (diagnostic superclasses)")
    database, labels, label_report = build_superclass_labels(database, statements, cfg)
    save_json(label_report, results_dir / "label_report.json")
    summary["labels"] = label_report
    summary["dataset_description"] = describe_dataset(database, labels, cfg)
    save_json(summary["dataset_description"], results_dir / "dataset_description.json")

    # -------------------------------------------------------------- 3. split
    log_section(logger, "3. official strat_fold split (patient-disjoint)")
    splits = split_by_fold(database, labels, cfg)
    split_summary = {name: splits[name]["summary"] for name in ("train", "val", "test")}
    split_summary["patient_overlap"] = splits["patient_overlap"]
    save_json(split_summary, results_dir / "split_summary.json")
    summary["split"] = split_summary

    # -------------------------------------------------------------- 4. cache
    log_section(logger, "4. waveform preprocessing and caching")
    waveforms, cache_path = build_waveform_cache(database, cfg, force=force_cache, root=root)
    summary["cache"] = {"path": str(cache_path), "shape": list(waveforms.shape)}

    y_train = splits["train"]["labels"]
    y_val = splits["val"]["labels"]
    y_test = splits["test"]["labels"]
    experiments: dict[str, dict[str, Any]] = {}

    # ----------------------------------------------------------- 5. baseline
    if not skip_baseline:
        log_section(logger, "5. classical baseline (Experiment E-A)")
        features = list(cfg.baseline.features)
        start = time.time()
        X_train = extract_feature_matrix(waveforms, splits["train"]["indices"], features)
        X_val = extract_feature_matrix(waveforms, splits["val"]["indices"], features)
        X_test = extract_feature_matrix(waveforms, splits["test"]["indices"], features)

        baseline = train_baseline(X_train, y_train, X_val, cfg, X_test=X_test)
        thresholds_baseline, _info = ev.tune_per_class_thresholds(
            y_val, baseline["val_prob"], cfg
        )
        metrics_baseline = ev.evaluate_multilabel(
            y_test, baseline["test_prob"], thresholds_baseline, cfg,
            split_name="test/baseline",
        )
        experiments["statistical_baseline"] = {
            "metrics": metrics_baseline,
            "parameters": int(X_train.shape[1] * len(classes)),
            "train_seconds": round(time.time() - start, 2),
            "thresholds": thresholds_baseline.tolist(),
        }
        save_pickle(baseline["model"], models_dir / "ecg_baseline.pkl")
        save_json({
            "features": features,
            "feature_names": feature_names(features, list(cfg.dataset.lead_names)),
            "n_features": int(X_train.shape[1]),
            "metrics": metrics_baseline,
        }, results_dir / "baseline_metrics.json")
    else:
        logger.info("Skipping the classical baseline (--skip-baseline).")

    # -------------------------------------------------------- 6. deep model
    log_section(logger, f"6. deep model: {cfg.model.name} (Experiment E-B/E-C)")
    loaders = build_dataloaders(waveforms, labels, splits, cfg)
    pos_weight = None
    if str(cfg.training.get("pos_weight", "auto")) == "auto":
        pos_weight = compute_pos_weight(y_train)

    model = build_model(cfg).to(device)
    architecture = model_summary(
        model, input_shape=(2, int(cfg.dataset.n_leads), int(cfg.dataset.signal_length))
    )
    logger.info("Architecture: %s", architecture)
    summary["architecture"] = architecture

    training_result = train_model(
        model, loaders, cfg, device, checkpoint_dir,
        pos_weight=pos_weight, experiment_name=str(cfg.model.name),
    )
    summary["training"] = training_result

    plot_training_curves(
        training_result["history"], results_dir / "training_curve.png",
        loss_keys=("train_loss", "val_loss"),
        metric_keys=("val_macro_auc", "val_macro_pr_auc", "val_macro_f1"),
        best_epoch=training_result["best_epoch"],
        title=f"{cfg.model.name} — training history",
    )

    # ----------------------------------------------------- 7. thresholds/eval
    log_section(logger, "7. per-class thresholds (validation) and test evaluation")
    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"

    val_prob, val_true = ev.predict_probabilities(model, loaders["val"], device, use_amp)
    thresholds, threshold_info = ev.tune_per_class_thresholds(val_true, val_prob, cfg)
    save_json(threshold_info, results_dir / "thresholds.json")

    test_prob, test_true = ev.predict_probabilities(model, loaders["test"], device, use_amp)
    test_metrics = ev.evaluate_multilabel(test_true, test_prob, thresholds, cfg,
                                          split_name="test")

    experiments[str(cfg.model.name)] = {
        "metrics": test_metrics,
        "parameters": count_parameters(model),
        "train_seconds": training_result["total_seconds"],
        "thresholds": thresholds.tolist(),
    }
    summary["test_metrics"] = test_metrics
    summary["thresholds"] = threshold_info

    per_class = ev.per_class_table(test_metrics, classes)
    save_dataframe(per_class, results_dir / "per_class_metrics.csv")
    logger.info("\n%s", per_class.to_string(index=False))

    ev.plot_per_class_curves(test_true, test_prob, classes, results_dir, prefix="test")
    ev.plot_per_class_confusion(test_true, test_prob, thresholds, classes, results_dir)

    comparison = ev.build_comparison_table(experiments)
    save_dataframe(comparison, results_dir / "model_comparison.csv")
    logger.info("\n%s", comparison.to_string(index=False))
    summary["model_comparison"] = comparison.to_dict(orient="records")

    # Does the deep model actually earn its complexity? The brief asks for this
    # verdict explicitly, and the honest answer is sometimes "no".
    recommendation = ev.recommend_model(experiments)
    save_json(recommendation, results_dir / "model_recommendation.json")
    summary["recommendation"] = recommendation

    # ----------------------------------------------------- 8. error analysis
    log_section(logger, "8. error analysis")
    errors = ev.export_errors(
        splits["test"]["database"], test_true, test_prob, thresholds, classes,
        results_dir / "errors", prefix="test",
    )
    summary["errors"] = errors

    # ------------------------------------------------------------- 9. IG/XAI
    if not skip_explain:
        log_section(logger, "9. explainability (Integrated Gradients)")
        try:
            summary["explanations"] = run_ig_analysis(
                model, waveforms, splits["test"]["indices"], test_true, test_prob,
                thresholds, cfg, device, results_dir / "explanations",
            )
        except Exception as exc:  # noqa: BLE001 - XAI must not lose a trained model
            logger.warning("Integrated Gradients failed (%s); model is still saved.", exc)
    else:
        logger.info("Skipping explainability (--skip-explain).")

    # ---------------------------------------------------------- 10. artifacts
    log_section(logger, "10. saving artifacts")
    model_path = models_dir / cfg.output.model_file
    torch.save({
        "model_state": model.state_dict(),
        "model_name": str(cfg.model.name),
        "class_names": classes,
        "thresholds": thresholds.tolist(),
    }, model_path)

    inference_config = {
        "model_version": str(cfg.output.model_version),
        "model_name": str(cfg.model.name),
        "model_params": cfg.model.get(str(cfg.model.name), {}).to_dict()
        if hasattr(cfg.model.get(str(cfg.model.name), {}), "to_dict") else {},
        "classes": classes,
        "label_mapping": {str(i): name for i, name in enumerate(classes)},
        "thresholds": {name: round(float(t), 4) for name, t in zip(classes, thresholds)},
        "task_type": str(cfg.task.type),
        "sampling_rate": int(cfg.dataset.sampling_rate),
        "signal_length": int(cfg.dataset.signal_length),
        "n_leads": int(cfg.dataset.n_leads),
        "lead_names": list(cfg.dataset.lead_names),
        "preprocessing": cfg.preprocessing.to_dict(),
    }
    config_path = save_json(inference_config, models_dir / cfg.output.config_file)

    metadata = {
        **inference_config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modality": "ecg",
        "dataset": {"name": str(cfg.dataset.name), "version": str(cfg.dataset.version),
                    "root": str(root)},
        "architecture": architecture,
        "split_summary": split_summary,
        "label_report": label_report,
        "training": training_result,
        "test_metrics": test_metrics,
        "threshold_info": threshold_info,
        "training_config": cfg.to_dict(),
    }
    metadata_path = save_json(metadata, models_dir / cfg.output.metadata_file)

    save_json({
        "model": str(cfg.model.name),
        "test": test_metrics,
        "model_comparison": comparison.to_dict(orient="records"),
        "recommendation": recommendation,
        "split": split_summary,
        "thresholds": threshold_info,
        "training": training_result,
    }, results_dir / "metrics.json")

    summary["artifacts"] = {
        "model": str(model_path), "config": str(config_path), "metadata": str(metadata_path),
        "checkpoints": str(checkpoint_dir),
    }
    summary["duration_sec"] = round(time.time() - started, 2)

    log_section(logger, "ecg pipeline complete")
    logger.info("macro ROC-AUC %.4f | macro F1 %.4f | best epoch %s | %d params",
                test_metrics["macro_roc_auc"], test_metrics["macro_f1"],
                training_result["best_epoch"], count_parameters(model))
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the CardioSense ECG pipeline.")
    parser.add_argument("--config", default="ecg", help="Config name or YAML path.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Override a config value.")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--force-cache", action="store_true",
                        help="Rebuild the waveform cache from the WFDB files.")
    parser.add_argument("--experiment-name", default="ecg_pipeline")
    args = parser.parse_args(argv)

    overrides = dict(_parse_override(item) for item in args.overrides)
    cfg = load_config(args.config, overrides=overrides)
    print_environment()

    with ExperimentTracker(args.experiment_name, modality="ecg", config=cfg,
                           primary_metric="macro_roc_auc") as run:
        summary = run_pipeline(
            cfg,
            skip_baseline=args.skip_baseline,
            skip_explain=args.skip_explain,
            force_cache=args.force_cache,
        )
        run.log_params({
            "model": str(cfg.model.name),
            "batch_size": int(cfg.training.batch_size),
            "learning_rate": float(cfg.training.learning_rate),
            "epochs": int(cfg.training.epochs),
            "parameters": summary["architecture"]["total"],
            "n_train": summary["split"]["train"]["n_records"],
            "sampling_rate": int(cfg.dataset.sampling_rate),
        })
        run.log_metrics(
            {k: v for k, v in summary["test_metrics"].items()
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
