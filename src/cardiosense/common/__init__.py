"""Shared utilities used by all three Phase 1 pipelines."""

from .compat import autocast_context, make_grad_scaler, make_prefit_calibrator
from .config import Config, load_config, save_config
from .env import describe_environment, get_device, is_colab, print_environment
from .experiment import ExperimentTracker, load_experiment_log
from .io_utils import (
    load_json,
    load_pickle,
    save_dataframe,
    save_json,
    save_pickle,
    timestamp,
)
from .logging_utils import configure_logging, get_logger, log_section
from .metrics import (
    binary_metrics,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    find_best_threshold,
    metrics_to_frame,
    multiclass_metrics,
    multilabel_metrics,
)
from .paths import PATHS, ProjectPaths, ensure_dir, resolve_path
from .plots import (
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
    save_figure,
)
from .seeding import get_generator, seed_worker, set_seed
from .training import AverageMeter, CheckpointManager, EarlyStopping, History, count_parameters

__all__ = [
    "Config", "load_config", "save_config",
    "PATHS", "ProjectPaths", "resolve_path", "ensure_dir",
    "set_seed", "seed_worker", "get_generator",
    "get_logger", "configure_logging", "log_section",
    "get_device", "describe_environment", "print_environment", "is_colab",
    "save_json", "load_json", "save_pickle", "load_pickle", "save_dataframe", "timestamp",
    "binary_metrics", "multiclass_metrics", "multilabel_metrics", "brier_score",
    "expected_calibration_error", "find_best_threshold", "bootstrap_ci", "metrics_to_frame",
    "plot_confusion_matrix", "plot_roc_curve", "plot_pr_curve", "plot_calibration_curve",
    "plot_training_curves", "plot_class_distribution", "save_figure",
    "ExperimentTracker", "load_experiment_log",
    "AverageMeter", "EarlyStopping", "CheckpointManager", "History", "count_parameters",
    "make_prefit_calibrator", "make_grad_scaler", "autocast_context",
]
