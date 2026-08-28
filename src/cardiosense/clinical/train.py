"""End-to-end training entry point for the clinical pipeline.

Run it::

    python -m cardiosense.clinical.train
    python -m cardiosense.clinical.train --config configs/clinical_config.yaml
    python -m cardiosense.clinical.train --set models.xgboost.n_iter_search=10 --skip-eda

The pipeline, in order:

    load -> clean -> EDA -> split -> fit preprocessor (train only) -> tune LogReg
    -> tune XGBoost -> select on validation -> tune threshold on validation
    -> choose calibration method by CV on train -> fit calibrator on validation
    -> score test ONCE -> error analysis -> SHAP -> save artifacts

Every stage writes to ``results/clinical/`` and the whole run is recorded by the
experiment tracker, so a result can always be traced back to its configuration,
git commit and hardware.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..common.config import Config, load_config
from ..common.experiment import ExperimentTracker
from ..common.io_utils import save_dataframe, save_json, save_pickle
from ..common.logging_utils import get_logger, log_section
from ..common.paths import PATHS, ensure_dir
from ..common.seeding import set_seed
from . import calibrate as calibration_module
from . import evaluate as evaluation
from .data import load_raw_dataframe, prepare_dataset
from .eda import run_eda
from .explain import run_shap_analysis
from .models import tune_logistic_regression, tune_xgboost
from .preprocessing import build_preprocessor, fit_preprocessor, split_data, transform_splits

__all__ = ["run_pipeline", "main"]

logger = get_logger("clinical.train")


def _parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value`` into a typed ``(key, value)`` pair."""
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
    skip_eda: bool = False,
    skip_shap: bool = False,
) -> dict[str, Any]:
    """Execute the full clinical pipeline and return a summary of every stage.

    Args:
        cfg: Clinical configuration.
        skip_eda: Skip EDA figure generation (useful when re-running training).
        skip_shap: Skip SHAP (useful if shap is unavailable).

    Returns:
        A summary dict containing metrics, decisions and artifact paths.
    """
    started = time.time()
    set_seed(int(cfg.seed), strict=bool(cfg.get("strict_determinism", True)))

    results_dir = ensure_dir(_resolve_out(cfg.output.results_dir))
    models_dir = ensure_dir(_resolve_out(cfg.output.models_dir))
    summary: dict[str, Any] = {"config_path": cfg.get("_config_path")}

    # ---------------------------------------------------------------- 1. data
    log_section(logger, "1. dataset")
    raw = load_raw_dataframe(cfg)
    X, y, data_report = prepare_dataset(raw, cfg)
    save_json(data_report, results_dir / "dataset_report.json")
    summary["dataset"] = data_report

    # ----------------------------------------------------------------- 2. EDA
    if not skip_eda:
        log_section(logger, "2. exploratory data analysis")
        summary["eda"] = run_eda(X, y, cfg, results_dir / "eda")
    else:
        logger.info("Skipping EDA (--skip-eda).")

    # --------------------------------------------------------------- 3. split
    log_section(logger, "3. train / validation / test split")
    splits = split_data(X, y, cfg)
    save_json(splits.summary, results_dir / "split_summary.json")
    summary["split"] = splits.summary

    # -------------------------------------------------------- 4. preprocessing
    log_section(logger, "4. preprocessing (fit on train only)")
    preprocessor = fit_preprocessor(build_preprocessor(cfg), splits.X_train, splits.y_train)
    X_train_t, X_val_t, X_test_t, feature_names = transform_splits(preprocessor, splits)
    y_train = splits.y_train.to_numpy()
    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()
    summary["features"] = {"n_encoded": len(feature_names), "names": feature_names}

    # ------------------------------------------------------------- 5. training
    log_section(logger, "5. model training and hyperparameter search")
    experiments: dict[str, dict[str, Any]] = {}

    if bool(cfg.models.logistic_regression.get("enabled", True)):
        search = tune_logistic_regression(X_train_t, y_train, cfg)
        model = search["estimator"]
        val_prob = evaluation.predict_proba(model, X_val_t)
        experiments["logistic_regression"] = {
            **{k: v for k, v in search.items() if k != "estimator"},
            "model": model,
            "val": evaluation.evaluate_at_threshold(y_val, val_prob, 0.5),
            "val_prob": val_prob,
        }

    if bool(cfg.models.xgboost.get("enabled", True)):
        search = tune_xgboost(X_train_t, y_train, cfg)
        model = search["estimator"]
        val_prob = evaluation.predict_proba(model, X_val_t)
        experiments["xgboost"] = {
            **{k: v for k, v in search.items() if k != "estimator"},
            "model": model,
            "val": evaluation.evaluate_at_threshold(y_val, val_prob, 0.5),
            "val_prob": val_prob,
        }

    if not experiments:
        raise RuntimeError("No models are enabled in the configuration.")

    # ------------------------------------------------------------ 6. selection
    log_section(logger, "6. model comparison and selection")
    comparison = evaluation.build_comparison_table(
        experiments, metrics=list(cfg.selection.get("report_metrics", []))
    )
    save_dataframe(comparison, results_dir / "model_comparison.csv")
    logger.info("\n%s", comparison.to_string(index=False))

    selected_name, decision = evaluation.select_model(experiments, cfg)
    decision["best_params"] = experiments[selected_name].get("best_params", {})
    save_json(decision, results_dir / "model_selection.json")
    selected_model = experiments[selected_name]["model"]
    summary["model_comparison"] = comparison.to_dict(orient="records")
    summary["selection"] = decision

    # ------------------------------------------------------------ 7. threshold
    log_section(logger, "7. operating threshold (tuned on validation)")
    val_prob = experiments[selected_name]["val_prob"]
    # Out-of-fold training predictions make the threshold estimate far more stable
    # than 45 validation points alone. See evaluate.tune_threshold for the argument.
    oof_prob = None
    if str(cfg.selection.get("threshold_tuning_data", "cv_oof_plus_val")) == "cv_oof_plus_val":
        from sklearn.base import clone

        oof_prob = evaluation.out_of_fold_probabilities(
            clone(selected_model), X_train_t, y_train, cfg,
            folds=int(cfg.models.xgboost.get("cv_folds", 5)),
        )
    threshold, threshold_info = evaluation.tune_threshold(
        y_val, val_prob, cfg, oof_y=y_train, oof_prob=oof_prob
    )
    summary["threshold"] = threshold_info

    # ---------------------------------------------------------- 8. calibration
    log_section(logger, "8. probability calibration")
    method, method_report = calibration_module.select_calibration_method(
        selected_model, X_train_t, y_train, cfg
    )
    calibrator = calibration_module.fit_calibrator(selected_model, X_val_t, y_val, method=method)
    save_json(method_report, results_dir / "calibration_method_selection.json")

    test_prob_raw = evaluation.predict_proba(selected_model, X_test_t)
    test_prob_cal = evaluation.predict_proba(calibrator, X_test_t)

    calibration_metrics = calibration_module.calibration_report(
        y_test, test_prob_raw, test_prob_cal, cfg, results_dir, method=method, split_name="test"
    )
    summary["calibration"] = {"method_selection": method_report, "metrics": calibration_metrics}

    # ----------------------------------------------------- 9. final evaluation
    log_section(logger, "9. final evaluation on the held-out test split")
    extra_curves = {
        name: (y_test, evaluation.predict_proba(blocks["model"], X_test_t))
        for name, blocks in experiments.items() if name != selected_name
    }
    test_metrics = evaluation.evaluate_final(
        selected_name, y_test, test_prob_raw, threshold, cfg, results_dir,
        extra_curves=extra_curves,
    )
    calibrated_test_metrics = evaluation.evaluate_at_threshold(y_test, test_prob_cal, threshold)
    test_metrics["calibrated_at_tuned_threshold"] = calibrated_test_metrics
    summary["test_metrics"] = test_metrics

    # ------------------------------------------------------ 10. error analysis
    log_section(logger, "10. error analysis")
    errors = evaluation.export_errors(
        splits.X_test, y_test, test_prob_raw, threshold,
        results_dir / "errors", prefix="test",
    )
    save_json(errors, results_dir / "errors" / "error_summary.json")
    summary["errors"] = errors

    # ---------------------------------------------------------------- 11. SHAP
    if not skip_shap:
        log_section(logger, "11. SHAP explanations")
        y_pred_test = (test_prob_raw >= threshold).astype(int)
        case_rows = {
            "TP": np.flatnonzero((y_test == 1) & (y_pred_test == 1))[:2].tolist(),
            "TN": np.flatnonzero((y_test == 0) & (y_pred_test == 0))[:1].tolist(),
            "FP": np.flatnonzero((y_test == 0) & (y_pred_test == 1))[:2].tolist(),
            "FN": np.flatnonzero((y_test == 1) & (y_pred_test == 0))[:2].tolist(),
        }
        case_rows = {k: v for k, v in case_rows.items() if v}
        try:
            summary["shap"] = run_shap_analysis(
                selected_model, X_train_t, X_test_t, feature_names, cfg,
                results_dir / "shap", case_selection=case_rows,
            )
        except ImportError as exc:
            logger.warning("SHAP unavailable (%s); skipping explanations.", exc)
    else:
        logger.info("Skipping SHAP (--skip-shap).")

    # ----------------------------------------------------------- 12. artifacts
    log_section(logger, "12. saving artifacts")
    model_path = save_pickle(selected_model, models_dir / cfg.output.model_file)
    preprocessor_path = save_pickle(preprocessor, models_dir / cfg.output.preprocessor_file)
    calibrator_path = save_pickle(calibrator, models_dir / cfg.output.calibrator_file)

    metadata = {
        "model_version": str(cfg.output.model_version),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modality": "clinical",
        "selected_model": selected_name,
        "selected_model_class": type(selected_model).__name__,
        "best_params": experiments[selected_name].get("best_params", {}),
        "dataset": {
            "name": str(cfg.dataset.name),
            "source": str(cfg.dataset.source),
            "target_column": str(cfg.dataset.target_column),
            "target_rule": data_report["target_rule"],
            "positive_class_meaning": str(cfg.dataset.get("positive_class_meaning", "")),
        },
        "raw_features": {
            "numeric": list(cfg.dataset.numeric_features),
            "categorical": list(cfg.dataset.categorical_features),
            "order": list(splits.X_train.columns),
        },
        "encoded_feature_names": feature_names,
        "label_mapping": {"0": "no disease", "1": "disease present"},
        "threshold": threshold,
        "threshold_info": threshold_info,
        "calibration": {"method": method, "fitted_on": "validation split",
                        "metrics": calibration_metrics},
        "split_summary": splits.summary,
        "test_metrics": test_metrics,
        "training_config": cfg.to_dict(),
        "artifacts": {
            "model": model_path.name,
            "preprocessor": preprocessor_path.name,
            "calibrator": calibrator_path.name,
        },
    }
    metadata_path = save_json(metadata, models_dir / cfg.output.metadata_file)

    # A compact metrics.json at the top of results/clinical/ for the report.
    save_json(
        {
            "model": selected_name,
            "test": test_metrics,
            "calibration": calibration_metrics,
            "model_comparison": comparison.to_dict(orient="records"),
            "selection": decision,
            "split": splits.summary,
        },
        results_dir / "metrics.json",
    )

    summary["artifacts"] = {
        "model": str(model_path), "preprocessor": str(preprocessor_path),
        "calibrator": str(calibrator_path), "metadata": str(metadata_path),
    }
    summary["duration_sec"] = round(time.time() - started, 2)

    log_section(logger, "clinical pipeline complete")
    logger.info("Selected: %s | test ROC-AUC %.3f | recall %.3f | Brier %.3f -> %.3f",
                selected_name, test_metrics["at_tuned_threshold"]["roc_auc"],
                test_metrics["at_tuned_threshold"]["recall"],
                calibration_metrics["uncalibrated"]["brier"],
                calibration_metrics["calibrated"]["brier"])
    logger.info("Artifacts in %s | results in %s", models_dir, results_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the CardioSense clinical pipeline.")
    parser.add_argument("--config", default="clinical",
                        help="Config name ('clinical') or path to a YAML file.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Override a config value, e.g. seed=7.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figures.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explanations.")
    parser.add_argument("--experiment-name", default="clinical_pipeline",
                        help="Name recorded in the experiment log.")
    args = parser.parse_args(argv)

    overrides = dict(_parse_override(item) for item in args.overrides)
    cfg = load_config(args.config, overrides=overrides)

    with ExperimentTracker(args.experiment_name, modality="clinical", config=cfg,
                           primary_metric="roc_auc") as run:
        summary = run_pipeline(cfg, skip_eda=args.skip_eda, skip_shap=args.skip_shap)

        selected = summary["selection"]["selected"]
        run.log_params({
            "model": selected,
            "best_params": summary["selection"].get("best_params", {}),
            "threshold": summary["threshold"]["threshold"],
            "calibration_method": summary["calibration"]["method_selection"]["selected"],
            "n_train": summary["split"]["sizes"]["train"],
            "n_val": summary["split"]["sizes"]["val"],
            "n_test": summary["split"]["sizes"]["test"],
        })
        run.log_metrics(summary["test_metrics"]["at_tuned_threshold"], split="test")
        run.log_metrics({"n_errors": summary["errors"]["n_errors"],
                         **{f"brier_{k}": v for k, v in
                            summary["calibration"]["metrics"]["improvement"].items()}},
                        split="calibration")
        for name, path in summary["artifacts"].items():
            run.log_artifact(name, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
