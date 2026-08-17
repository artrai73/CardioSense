"""Runtime environment inspection.

Two jobs:

1. :func:`get_device` — the single place that decides CPU vs CUDA vs MPS.
2. :func:`describe_environment` — a JSON-serialisable fingerprint of the machine,
   recorded with every experiment so results can be attributed to hardware.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

__all__ = ["get_device", "describe_environment", "print_environment", "is_colab", "gpu_summary"]


def is_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules or _module_available("google.colab")


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def get_device(prefer: str = "auto", verbose: bool = True) -> Any:
    """Return the ``torch.device`` to train on.

    Args:
        prefer: ``"auto"`` (default), ``"cuda"``, ``"mps"`` or ``"cpu"``.
        verbose: Print the selected device and GPU details.

    Returns:
        A ``torch.device``.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for get_device(). pip install torch") from exc

    if prefer != "auto":
        device = torch.device(prefer)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"[env] device: {device}")
        if device.type == "cuda":
            index = device.index or 0
            props = torch.cuda.get_device_properties(index)
            print(f"[env] gpu   : {props.name}")
            print(f"[env] memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"[env] cuda  : {torch.version.cuda} | capability sm_{props.major}{props.minor}")
        elif device.type == "cpu":
            print("[env] WARNING: no GPU detected. On Colab: Runtime > Change runtime "
                  "type > Hardware accelerator > GPU.")
    return device


def gpu_summary() -> dict[str, Any]:
    """Return GPU details, or ``{"available": False}`` when there is no CUDA GPU."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda not available"}

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "capability": f"sm_{props.major}{props.minor}",
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def _git_commit() -> str | None:
    """Short git SHA of the working tree, or ``None`` outside a repo."""
    try:
        from .paths import PATHS

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PATHS.root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str]:
    """Versions of the packages that actually affect numerical results."""
    versions: dict[str, str] = {}
    for module_name in ("numpy", "pandas", "sklearn", "scipy", "xgboost", "shap",
                        "torch", "torchvision", "wfdb", "captum"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def describe_environment() -> dict[str, Any]:
    """Return a JSON-serialisable fingerprint of the runtime.

    Recorded in every experiment log so that a result can always be traced back
    to the exact hardware and library versions that produced it.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": _cpu_count(),
        "is_colab": is_colab(),
        "gpu": gpu_summary(),
        "git_commit": _git_commit(),
        "packages": _package_versions(),
    }


def _cpu_count() -> int:
    import os

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def print_environment() -> dict[str, Any]:
    """Pretty-print :func:`describe_environment` and return it."""
    info = describe_environment()
    print("=" * 70)
    print("CARDIOSENSE RUNTIME ENVIRONMENT")
    print("=" * 70)
    print(f"  Python      : {info['python']}")
    print(f"  Platform    : {info['platform']}")
    print(f"  CPU cores   : {info['cpu_count']}")
    print(f"  Colab       : {info['is_colab']}")
    print(f"  Git commit  : {info['git_commit'] or 'n/a'}")
    gpu = info["gpu"]
    if gpu.get("available"):
        print(f"  GPU         : {gpu['name']} ({gpu['total_memory_gb']} GB, {gpu['capability']})")
        print(f"  CUDA        : {gpu['cuda_version']}")
    else:
        print(f"  GPU         : none ({gpu.get('reason')})")
    print("  Packages    :")
    for name, version in sorted(info["packages"].items()):
        print(f"      {name:<14} {version}")
    print("=" * 70)
    return info
