"""CardioSense chest X-ray pipeline — cardiomegaly detection on NIH ChestX-ray14.

Binary target, heavily imbalanced. PR-AUC is the headline metric; accuracy is
reported only alongside the majority-class baseline so it cannot be misread.

Pipeline order::

    data.verify_dataset -> data.load_metadata -> data.build_target
    -> data.split_by_patient -> preprocessing.build_transforms
    -> baseline.* -> models.build_model -> trainer.train_model
    -> evaluate.* -> explain.run_gradcam_analysis

Entry points::

    python -m cardiosense.xray.train
    python -m cardiosense.xray.predict --image <path>
"""

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
from .evaluate import evaluate_binary, predict_probabilities, select_error_examples, tune_threshold
from .explain import GradCAM, compute_gradcam, run_gradcam_analysis
from .models import XrayDenseNet121, build_model, model_summary, set_backbone_trainable
from .predict import XrayPredictor
from .preprocessing import build_eval_transform, build_train_transform, build_transforms
from .trainer import train_model

__all__ = [
    "verify_dataset", "load_metadata", "build_target", "split_by_patient",
    "describe_dataset", "resolve_nih_root",
    "build_transforms", "build_train_transform", "build_eval_transform",
    "ChestXrayDataset", "build_dataloaders", "compute_pos_weight",
    "majority_class_baseline", "train_pixel_baseline", "build_feature_matrix",
    "XrayDenseNet121", "build_model", "set_backbone_trainable", "model_summary",
    "train_model",
    "predict_probabilities", "tune_threshold", "evaluate_binary", "select_error_examples",
    "GradCAM", "compute_gradcam", "run_gradcam_analysis",
    "XrayPredictor",
]
