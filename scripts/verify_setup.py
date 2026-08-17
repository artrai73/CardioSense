#!/usr/bin/env python
"""Verify that a CardioSense environment is ready to train.

Run this first, on every machine, before anything else::

    python scripts/verify_setup.py
    python scripts/verify_setup.py --check-data      # also probe the datasets

It checks, and reports pass/warn/fail for:

* Python version and the project package being importable
* Required and optional third-party packages
* GPU availability and CUDA version
* Project directory layout (creating anything missing)
* All three YAML configs parsing correctly
* Optionally, whether each dataset is present where the configs expect it

Exit code is 0 when nothing FAILED (warnings are tolerated), 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Allow running before `pip install -e .`
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cardiosense.common.config import load_config  # noqa: E402
from cardiosense.common.env import print_environment  # noqa: E402
from cardiosense.common.paths import PATHS, resolve_path  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {PASS: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}

REQUIRED_PACKAGES = ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "yaml", "joblib"]
OPTIONAL_PACKAGES = {
    "torch": "deep learning (ECG + X-ray)",
    "torchvision": "DenseNet121 weights (X-ray)",
    "xgboost": "clinical advanced model",
    "shap": "clinical explainability",
    "captum": "ECG integrated gradients",
    "wfdb": "PTB-XL waveform reading",
    "ucimlrepo": "automatic UCI Heart Disease download",
    "cv2": "Grad-CAM overlays",
    "seaborn": "EDA figures",
}

_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"  {_SYMBOL[status]} {name}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        record(PASS, f"Python {major}.{minor}")
    else:
        record(FAIL, f"Python {major}.{minor}", "CardioSense requires Python >= 3.10")


def check_package_imports() -> None:
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            record(PASS, name, getattr(module, "__version__", ""))
        except ImportError:
            record(FAIL, name, "required — pip install -r requirements.txt")

    for name, purpose in OPTIONAL_PACKAGES.items():
        try:
            module = importlib.import_module(name)
            record(PASS, name, getattr(module, "__version__", "") or purpose)
        except ImportError:
            record(WARN, name, f"missing — needed for {purpose}")


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        record(WARN, "GPU", "torch not installed; CPU-only workflows still run")
        return

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record(PASS, "GPU", f"{props.name}, {props.total_memory / 1024**3:.1f} GB, "
                            f"CUDA {torch.version.cuda}")
    else:
        record(WARN, "GPU", "no CUDA device. On Colab: Runtime > Change runtime type > GPU")


def check_directories() -> None:
    PATHS.ensure_all()
    for name in ("root", "configs", "data", "models", "results"):
        path = getattr(PATHS, name)
        record(PASS if path.exists() else FAIL, f"dir:{name}", str(path))


def check_configs() -> None:
    for modality in ("clinical", "ecg", "xray"):
        try:
            cfg = load_config(modality)
            record(PASS, f"config:{modality}", f"seed={cfg.seed}, modality={cfg.modality}")
        except Exception as exc:  # noqa: BLE001 - report any parse problem
            record(FAIL, f"config:{modality}", f"{type(exc).__name__}: {exc}")


def _resolve_data(value: str) -> Path:
    """Resolve a config path against the data root when it starts with 'data/'."""
    text = str(value)
    if text.startswith("data/"):
        return (PATHS.data / text[len("data/"):]).resolve()
    return resolve_path(text)


def check_datasets() -> None:
    # --- Clinical ---------------------------------------------------------
    cfg = load_config("clinical")
    cached = _resolve_data(cfg.dataset.raw_cache_path)
    if cached.exists():
        record(PASS, "data:clinical", f"cached CSV at {cached}")
    else:
        try:
            importlib.import_module("ucimlrepo")
            record(WARN, "data:clinical",
                   "not downloaded yet — the notebook fetches it automatically via ucimlrepo")
        except ImportError:
            record(WARN, "data:clinical", "no cache and ucimlrepo missing; see docs/datasets.md")

    # --- ECG --------------------------------------------------------------
    cfg = load_config("ecg")
    root = _resolve_data(cfg.dataset.root)
    database = root / cfg.dataset.database_csv
    records_dir = root / f"records{cfg.dataset.sampling_rate}"
    if database.exists() and records_dir.exists():
        n_folders = len(list(records_dir.glob("*")))
        record(PASS, "data:ecg", f"PTB-XL at {root} ({n_folders} record folders)")
    elif root.exists():
        record(WARN, "data:ecg", f"{root} exists but {database.name} or "
                                 f"{records_dir.name}/ is missing")
    else:
        record(WARN, "data:ecg", f"PTB-XL not found at {root} — see docs/datasets.md")

    # --- X-ray ------------------------------------------------------------
    cfg = load_config("xray")
    root = _resolve_data(cfg.dataset.root)
    metadata = root / cfg.dataset.metadata_csv
    images = root / cfg.dataset.images_dir
    if metadata.exists() and images.exists():
        record(PASS, "data:xray", f"NIH ChestX-ray14 at {root}")
    elif root.exists():
        record(WARN, "data:xray", f"{root} exists but {metadata.name} or "
                                  f"{images.name}/ is missing")
    else:
        record(WARN, "data:xray", f"ChestX-ray14 not found at {root} — see docs/datasets.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the CardioSense environment.")
    parser.add_argument("--check-data", action="store_true",
                        help="Also probe for the three datasets on disk.")
    args = parser.parse_args()

    print_environment()

    print("\nPROJECT PATHS")
    print(PATHS.describe())

    print("\n1. Python")
    check_python()
    print("\n2. Packages")
    check_package_imports()
    print("\n3. Hardware")
    check_gpu()
    print("\n4. Directories")
    check_directories()
    print("\n5. Configuration files")
    check_configs()
    if args.check_data:
        print("\n6. Datasets")
        check_datasets()

    n_fail = sum(1 for status, _, _ in _results if status == FAIL)
    n_warn = sum(1 for status, _, _ in _results if status == WARN)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(_results) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failures")
    print("=" * 70)
    if n_fail:
        print("Fix the FAIL items before training. Warnings are usually fine at this stage.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
