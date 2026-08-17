"""Version-compatibility shims.

Two APIs used by this project changed recently. Rather than pinning old versions
(which breaks Colab, where you do not control the base image), the differences
are isolated here.

1. **scikit-learn calibration.** ``CalibratedClassifierCV(cv="prefit")`` was
   deprecated in scikit-learn 1.6 in favour of wrapping the fitted estimator in
   ``sklearn.frozen.FrozenEstimator``. :func:`make_prefit_calibrator` picks the
   right form for the installed version.

2. **torch AMP.** ``torch.cuda.amp.GradScaler`` / ``autocast`` were deprecated in
   torch 2.4 in favour of ``torch.amp.GradScaler("cuda")`` /
   ``torch.amp.autocast("cuda")``. :func:`make_grad_scaler` and
   :func:`autocast_context` handle both.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

__all__ = [
    "make_prefit_calibrator",
    "make_grad_scaler",
    "autocast_context",
    "sklearn_version",
    "torch_version",
]


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse ``"2.13.0+cu130"`` or ``"1.8.0rc1"`` into ``(2, 13, 0)`` / ``(1, 8, 0)``.

    Local build tags (``+cu130``) and pre-release suffixes are stripped first,
    otherwise the patch component picks up the CUDA version digits.
    """
    cleaned = version.split("+")[0].split("-")[0]
    parts: list[int] = []
    for chunk in cleaned.split(".")[:3]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def sklearn_version() -> tuple[int, ...]:
    import sklearn

    return _version_tuple(sklearn.__version__)


def torch_version() -> tuple[int, ...]:
    import torch

    return _version_tuple(torch.__version__)


def make_prefit_calibrator(estimator: Any, method: str = "sigmoid") -> Any:
    """Return a ``CalibratedClassifierCV`` that calibrates an ALREADY-FITTED model.

    The returned object must then be ``.fit()`` on a held-out calibration split
    (we use the validation split). The base estimator is not refit.

    Args:
        estimator: A fitted classifier exposing ``predict_proba``.
        method: ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.

    Returns:
        An unfitted ``CalibratedClassifierCV`` wrapping *estimator*.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if sklearn_version() >= (1, 6):
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    return CalibratedClassifierCV(estimator, method=method, cv="prefit")


def make_grad_scaler(enabled: bool = True, device_type: str = "cuda") -> Any:
    """Return a mixed-precision ``GradScaler`` appropriate to the torch version.

    When *enabled* is ``False`` or there is no CUDA device, a disabled scaler is
    returned so the training loop needs no branching.
    """
    import torch

    use = bool(enabled and device_type == "cuda" and torch.cuda.is_available())

    if torch_version() >= (2, 4):
        return torch.amp.GradScaler(device_type if use else "cpu", enabled=use)
    return torch.cuda.amp.GradScaler(enabled=use)


def autocast_context(enabled: bool = True, device_type: str = "cuda", dtype: Any = None) -> Any:
    """Return an ``autocast`` context manager, or a no-op when disabled."""
    import torch

    use = bool(enabled and device_type == "cuda" and torch.cuda.is_available())
    if not use:
        return nullcontext()

    if torch_version() >= (2, 4):
        return torch.amp.autocast(device_type, dtype=dtype or torch.float16)
    return torch.cuda.amp.autocast(dtype=dtype or torch.float16)
