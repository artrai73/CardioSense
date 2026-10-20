"""Confidence-aware multimodal fusion for CardioSense."""

from .schema import (
    ModalityPrediction, RISK_DEFINITIONS, RISK_THRESHOLDS, RELIABILITY_BANDS,
    UNCALIBRATED_RELIABILITY, decisiveness, prior_correct,
    reliability_from_ece, reliability_label, risk_level,
)
from .strategies import (
    STRATEGIES, ConfidenceLogOddsFusion, ConfidenceWeightedFusion,
    FusionOutput, MaxFusion, MeanFusion, NoisyOrFusion, build_strategy,
)
from .disagreement import (
    DisagreementReport, compute_agreement, compute_reliability_indicator,
    detect_disagreement,
)
from .adapters import (
    BaseAdapter, ClinicalAdapter, ECGAdapter, ECGJointAdapter, ECGSignalAdapter,
    XrayAdapter, load_all_adapters, resolve_ecg_adapter,
)
from .ecg_joint import (
    ECG_CLASSES, JointCheckpoint, build_joint_net, find_joint_checkpoints,
    load_joint_checkpoint, load_joint_ensemble, mean_logit, predict_joint,
    predict_signal_only,
)
from .engine import AssessmentResult, FusionEngine, MODALITIES, CLINICIAN_NOTICE, NOT_FOR

__all__ = [
    "ModalityPrediction", "RISK_DEFINITIONS", "RISK_THRESHOLDS", "RELIABILITY_BANDS",
    "UNCALIBRATED_RELIABILITY", "decisiveness", "prior_correct",
    "reliability_from_ece", "reliability_label", "risk_level",
    "STRATEGIES", "ConfidenceLogOddsFusion", "ConfidenceWeightedFusion",
    "FusionOutput", "MaxFusion", "MeanFusion", "NoisyOrFusion", "build_strategy",
    "DisagreementReport", "compute_agreement", "compute_reliability_indicator",
    "detect_disagreement",
    "BaseAdapter", "ClinicalAdapter", "ECGAdapter", "ECGJointAdapter",
    "ECGSignalAdapter", "XrayAdapter", "load_all_adapters", "resolve_ecg_adapter",
    "ECG_CLASSES", "JointCheckpoint", "build_joint_net", "find_joint_checkpoints",
    "load_joint_checkpoint", "load_joint_ensemble", "mean_logit", "predict_joint",
    "predict_signal_only",
    "AssessmentResult", "FusionEngine", "MODALITIES", "CLINICIAN_NOTICE", "NOT_FOR",
]
