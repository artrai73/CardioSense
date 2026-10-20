"""Ways of combining per-modality risk scores into one number.

Late fusion: each modality reaches its own decision, then the decisions are
combined. Early fusion (concatenating raw inputs) is impossible here -- a
13-value tabular vector, a 12x1000 waveform and a 224x224 image share no input
space. Joint fusion (concatenating learned representations) would need paired
training data this project does not have.

Late fusion is also the right choice for reasons beyond necessity: each modality
keeps its own preprocessing and explainability, a missing modality is dropped
from a sum rather than imputed into a fixed-width vector, and each contribution
stays inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .schema import ModalityPrediction

_EPS = 1e-6


def _logit(p: float) -> float:
    p = float(np.clip(p, _EPS, 1 - _EPS))
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


@dataclass
class FusionOutput:
    """The fused result, with the weights that produced it."""

    risk_score: float | None
    weights: dict[str, float]
    strategy: str
    basis: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": None if self.risk_score is None else round(float(self.risk_score), 5),
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "strategy": self.strategy,
            "fusion_basis": self.basis,
        }


def _usable(predictions: Sequence[ModalityPrediction]) -> list[ModalityPrediction]:
    return [p for p in predictions
            if p.available and p.risk_score is not None and np.isfinite(p.risk_score)]


class FusionStrategy:
    """Base class. Subclasses implement :meth:`_combine` over usable modalities."""

    name = "base"

    def fuse(self, predictions: Sequence[ModalityPrediction]) -> FusionOutput:
        usable = _usable(predictions)
        if not usable:
            # No modality available is a real, reportable state -- not an error,
            # and emphatically not a reason to invent a number.
            return FusionOutput(None, {}, self.name, [])
        if len(usable) == 1:
            only = usable[0]
            return FusionOutput(float(only.risk_score), {only.modality: 1.0},
                                self.name, [only.modality])
        return self._combine(usable)

    def _combine(self, usable: list[ModalityPrediction]) -> FusionOutput:
        raise NotImplementedError


class MeanFusion(FusionStrategy):
    """Unweighted average of risk scores. The baseline everything is measured against."""

    name = "mean"

    def _combine(self, usable):
        w = 1.0 / len(usable)
        score = float(np.mean([p.risk_score for p in usable]))
        return FusionOutput(score, {p.modality: w for p in usable},
                            self.name, [p.modality for p in usable])


class ConfidenceWeightedFusion(FusionStrategy):
    """Weighted mean, weights proportional to ``(reliability x decisiveness)^gamma``."""

    name = "confidence_weighted"

    def __init__(self, gamma: float = 1.0):
        self.gamma = float(gamma)

    def _combine(self, usable):
        raw = np.array([max(p.confidence, _EPS) ** self.gamma for p in usable])
        if raw.sum() <= 0:                       # every modality maximally unsure
            raw = np.ones(len(usable))
        w = raw / raw.sum()
        score = float(np.sum(w * np.array([p.risk_score for p in usable])))
        return FusionOutput(score, {p.modality: float(wi) for p, wi in zip(usable, w)},
                            self.name, [p.modality for p in usable])


class ConfidenceLogOddsFusion(FusionStrategy):
    """Confidence-weighted pooling in log-odds space. The CardioSense default.

    Probabilities do not live on a linear scale; log-odds is where independent
    evidence adds. Averaging probabilities directly is systematically less
    extreme than any input, which is why mean fusion under-commits and loses
    calibration even when every input is well calibrated.
    """

    name = "confidence_logodds"

    def __init__(self, gamma: float = 1.0):
        self.gamma = float(gamma)

    def _combine(self, usable):
        raw = np.array([max(p.confidence, _EPS) ** self.gamma for p in usable])
        if raw.sum() <= 0:
            raw = np.ones(len(usable))
        w = raw / raw.sum()
        pooled = float(np.sum(w * np.array([_logit(p.risk_score) for p in usable])))
        return FusionOutput(_sigmoid(pooled),
                            {p.modality: float(wi) for p, wi in zip(usable, w)},
                            self.name, [p.modality for p in usable])


class MaxFusion(FusionStrategy):
    """Take the highest risk. Deliberately conservative; included as a reference."""

    name = "max"

    def _combine(self, usable):
        idx = int(np.argmax([p.risk_score for p in usable]))
        return FusionOutput(float(usable[idx].risk_score),
                            {p.modality: (1.0 if i == idx else 0.0)
                             for i, p in enumerate(usable)},
                            self.name, [p.modality for p in usable])


class NoisyOrFusion(FusionStrategy):
    """``1 - prod(1 - p)``. Included because it fails instructively.

    Noisy-OR assumes conditionally independent detectors. Three views of the
    same heart are not independent, so it over-counts shared evidence and
    produces a mean predicted risk far above the base rate -- high AUC, terrible
    calibration. Keeping it in the comparison makes the point that a strategy
    cannot be chosen on discrimination alone.
    """

    name = "noisy_or"

    def _combine(self, usable):
        product = float(np.prod([1.0 - p.risk_score for p in usable]))
        w = 1.0 / len(usable)
        return FusionOutput(1.0 - product, {p.modality: w for p in usable},
                            self.name, [p.modality for p in usable])


STRATEGIES: dict[str, type[FusionStrategy]] = {
    "mean": MeanFusion,
    "max": MaxFusion,
    "noisy_or": NoisyOrFusion,
    "confidence_weighted": ConfidenceWeightedFusion,
    "confidence_logodds": ConfidenceLogOddsFusion,
}


def build_strategy(name: str = "confidence_logodds", **kwargs) -> FusionStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Available: {sorted(STRATEGIES)}")
    cls = STRATEGIES[name]
    if name in {"confidence_weighted", "confidence_logodds"}:
        return cls(**kwargs)
    return cls()
