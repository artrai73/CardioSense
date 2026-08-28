"""CardioSense clinical pipeline — tabular cardiovascular risk prediction.

Pipeline order::

    data.load_raw_dataframe -> data.prepare_dataset -> eda.run_eda
    -> preprocessing.split_data -> preprocessing.build_preprocessor
    -> models.tune_* -> evaluate.select_model -> calibrate.* -> explain.*

Entry points::

    python -m cardiosense.clinical.train
    python -m cardiosense.clinical.predict --example
"""

from .calibrate import calibrated_confidence, calibration_report, fit_calibrator
from .data import FEATURE_DESCRIPTIONS, load_raw_dataframe, prepare_dataset
from .eda import run_eda
from .evaluate import evaluate_final, export_errors, select_model, tune_threshold
from .explain import run_shap_analysis
from .models import build_logistic_regression, build_xgboost, tune_logistic_regression, tune_xgboost
from .preprocessing import DataSplits, build_preprocessor, fit_preprocessor, split_data
from .predict import ClinicalPredictor

__all__ = [
    "load_raw_dataframe", "prepare_dataset", "FEATURE_DESCRIPTIONS",
    "run_eda",
    "split_data", "build_preprocessor", "fit_preprocessor", "DataSplits",
    "build_logistic_regression", "build_xgboost",
    "tune_logistic_regression", "tune_xgboost",
    "select_model", "tune_threshold", "evaluate_final", "export_errors",
    "fit_calibrator", "calibration_report", "calibrated_confidence",
    "run_shap_analysis",
    "ClinicalPredictor",
]
