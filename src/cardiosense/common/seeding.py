"""Reproducibility helpers.

Every training entry point calls :func:`set_seed` before touching data.

On determinism (recorded here because the report must discuss the trade-off):

* Seeding Python / NumPy / PyTorch fixes weight initialisation, dropout masks,
  shuffling and augmentation draws. This alone makes runs reproducible on the
  *same* hardware for CPU work and for most GPU work.
* ``strict=True`` additionally sets ``torch.use_deterministic_algorithms(True)``,
  disables cuDNN benchmarking and sets ``CUBLAS_WORKSPACE_CONFIG``. GPU results
  then match bit-for-bit across runs, but throughput drops (typically 10-30% for
  the CNNs used here) because cuDNN can no longer autotune, and PyTorch raises a
  ``RuntimeError`` if an op has no deterministic kernel.
* Recommendation used in this project: ``strict=True`` for the clinical pipeline
  (free), ``strict=False`` for ECG/X-ray, with seed-variation quantified by
  re-running with 3 seeds and reporting mean +/- std.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["set_seed", "seed_worker", "get_generator"]


def set_seed(seed: int = 42, strict: bool = False, verbose: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Args:
        seed: The seed value.
        strict: Enable deterministic GPU algorithms. See module docstring.
        verbose: Print a one-line confirmation.

    Returns:
        The seed that was applied, so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        if verbose:
            print(f"[seed] python+numpy seeded with {seed} (torch not installed)")
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        # Required by some cuBLAS GEMM kernels for deterministic behaviour.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover
            print(f"[seed] could not enable full determinism: {exc}")
    else:
        torch.backends.cudnn.deterministic = False
        # Autotuning pays off because our input shapes are fixed across batches.
        torch.backends.cudnn.benchmark = True

    if verbose:
        mode = "strict/deterministic" if strict else "seeded (cudnn.benchmark on)"
        print(f"[seed] seed={seed} mode={mode}")
    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`.

    Without this, every worker process inherits the same NumPy seed and any
    NumPy-based augmentation repeats identically across workers.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 42) -> Any:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
