"""CardioSense ECG pipeline — 12-lead ECG interpretation on PTB-XL.

Task: 5-class diagnostic superclass classification (NORM, MI, STTC, CD, HYP),
**multi-label** — a single record can legitimately carry several superclasses.

Pipeline order::

    data.verify_dataset -> data.load_metadata -> data.build_superclass_labels
    -> data.split_by_fold -> preprocessing.build_waveform_cache
    -> baseline.train_baseline -> models.build_model -> trainer.train_model
    -> evaluate.* -> explain.run_ig_analysis

Entry points::

    python -m cardiosense.ecg.train
    python -m cardiosense.ecg.predict --record <path/to/record>
"""

from .baseline import extract_features, train_baseline
from .data import (
    SUPERCLASSES,
    build_superclass_labels,
    load_metadata,
    parse_scp_codes,
    resolve_ptbxl_root,
    split_by_fold,
    verify_dataset,
)
from .dataset import ECGDataset, build_dataloaders, compute_pos_weight
from .evaluate import evaluate_multilabel, predict_probabilities, tune_per_class_thresholds
from .explain import integrated_gradients, run_ig_analysis
from .models import ECGCNN1D, ECGResNet1D, build_model, model_summary
from .predict import ECGPredictor
from .preprocessing import build_waveform_cache, normalize_signal, preprocess_signal
from .trainer import train_model

__all__ = [
    "SUPERCLASSES", "verify_dataset", "load_metadata", "parse_scp_codes",
    "build_superclass_labels", "split_by_fold", "resolve_ptbxl_root",
    "preprocess_signal", "normalize_signal", "build_waveform_cache",
    "ECGDataset", "build_dataloaders", "compute_pos_weight",
    "extract_features", "train_baseline",
    "ECGCNN1D", "ECGResNet1D", "build_model", "model_summary",
    "train_model",
    "predict_probabilities", "tune_per_class_thresholds", "evaluate_multilabel",
    "integrated_gradients", "run_ig_analysis",
    "ECGPredictor",
]
