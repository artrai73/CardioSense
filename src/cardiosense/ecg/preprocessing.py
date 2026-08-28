"""ECG signal preprocessing and waveform caching.

Every operation here is justified, because "standard ECG preprocessing" is not a
justification and a filter applied for no reason can destroy the exact morphology
the model needs.

What is applied, and why
------------------------

**0.5 Hz high-pass (baseline wander removal) — ON.**
Respiration, electrode impedance drift and patient movement produce a slow
wandering baseline below about 0.5 Hz. It is not diagnostic, it varies wildly
between recordings, and it corrupts per-lead normalisation: a record with a large
drift gets a large standard deviation, so z-scoring shrinks its genuine QRS
amplitudes. Removed with a 3rd-order Butterworth filter applied via ``filtfilt``,
which is **zero-phase** — a causal filter would shift the ST segment in time,
which is precisely what the STTC class is about.

**Low-pass — OFF.**
PTB-XL is already band-limited by the recording hardware, and the 100 Hz files are
anti-alias filtered to below 50 Hz. Adding another low-pass would attenuate the
high-frequency content of the QRS complex — the sharp upstrokes that distinguish
conduction disturbances (the CD class). There is no noise left for it to remove
that is worth that cost.

**Powerline notch — OFF.**
Mains interference was handled upstream by the recording equipment. A 50/60 Hz
notch at a 100 Hz sampling rate would sit at or beyond Nyquist and is meaningless
here.

**Per-lead z-score normalisation — ON.**
Each lead of each record is standardised using **its own** mean and standard
deviation. Two consequences worth stating explicitly:

* It removes inter-patient amplitude variation caused by electrode placement,
  body habitus and skin impedance, which is nuisance variation, not signal.
* Because the statistics come from within a single record, **no training-set
  statistic touches validation or test**. This normalisation is leakage-free by
  construction, unlike a dataset-wide scaler.

The cost is that absolute voltage information is lost. That matters for
hypertrophy (HYP), which is partly an amplitude criterion — noted as a limitation
rather than glossed over. ``per_record_zscore`` is available as an ablation, which
preserves relative amplitude *between* leads.

**Resampling — OFF.**
PTB-XL ships native 100 Hz files. We read those directly rather than downsampling
the 500 Hz versions, so no resampling artefacts are introduced at all.

**Artefact clipping — ON (8 sigma).**
Electrode pops produce single-sample spikes tens of standard deviations high.
Left alone they dominate the loss. Clipping at 8 sigma after normalisation caps
them without touching real QRS peaks, which sit around 3-5 sigma.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from ..common.config import Config
from ..common.io_utils import save_json
from ..common.logging_utils import get_logger
from ..common.paths import PATHS, ensure_dir
from .data import resolve_ptbxl_root

__all__ = [
    "design_highpass",
    "remove_baseline_wander",
    "normalize_signal",
    "preprocess_signal",
    "load_raw_record",
    "build_waveform_cache",
    "load_waveform_cache",
]

logger = get_logger("ecg.preprocessing")


def _resolve_cache_dir(cfg: Config) -> Path:
    value = str(cfg.preprocessing.get("cache_path", "data/ecg/cache"))
    if value.startswith("data/"):
        return ensure_dir(PATHS.data / value[len("data/"):])
    path = Path(value).expanduser()
    return ensure_dir(path if path.is_absolute() else PATHS.root / path)


def design_highpass(cutoff_hz: float, sampling_rate: int, order: int = 3) -> tuple[Any, Any]:
    """Design a Butterworth high-pass filter in second-order-section form.

    SOS rather than transfer-function coefficients: at these low normalised
    cutoffs (0.5 Hz against a 100 Hz rate, i.e. 0.01 of Nyquist) a direct-form
    ``b, a`` implementation is numerically unstable and can produce NaNs.
    """
    nyquist = 0.5 * sampling_rate
    normalised = cutoff_hz / nyquist
    if not 0 < normalised < 1:
        raise ValueError(
            f"High-pass cutoff {cutoff_hz} Hz is invalid for a {sampling_rate} Hz signal."
        )
    return scipy_signal.butter(order, normalised, btype="highpass", output="sos")


def remove_baseline_wander(
    waveform: np.ndarray,
    sampling_rate: int,
    cutoff_hz: float = 0.5,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase high-pass filter to remove baseline drift.

    Args:
        waveform: Shape ``(n_leads, n_samples)``.
        sampling_rate: Hz.
        cutoff_hz: High-pass cutoff.
        order: Butterworth order.

    Returns:
        The filtered waveform, same shape and dtype ``float32``.
    """
    sos = design_highpass(cutoff_hz, sampling_rate, order)
    # filtfilt runs the filter forwards and backwards, so the net phase shift is
    # exactly zero. A one-directional filter would delay the ST segment.
    filtered = scipy_signal.sosfiltfilt(sos, waveform, axis=-1)
    return np.ascontiguousarray(filtered, dtype=np.float32)


def normalize_signal(
    waveform: np.ndarray,
    method: str = "per_lead_zscore",
    clip_sigma: float | None = 8.0,
) -> np.ndarray:
    """Normalise a waveform.

    Args:
        waveform: Shape ``(n_leads, n_samples)``.
        method: ``per_lead_zscore`` (each lead standardised independently),
            ``per_record_zscore`` (one mean/std for the whole record, preserving
            relative amplitude between leads), or ``none``.
        clip_sigma: Clip at this many standard deviations after normalising, to
            cap electrode-pop artefacts. ``None`` disables clipping.

    Returns:
        Normalised ``float32`` waveform.
    """
    x = np.asarray(waveform, dtype=np.float32)

    if method == "per_lead_zscore":
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
    elif method == "per_record_zscore":
        mean = x.mean(keepdims=True)
        std = x.std(keepdims=True)
    elif method in {"none", "null"}:
        return x
    else:
        raise ValueError(f"Unknown normalization {method!r}.")

    # A flat lead (disconnected electrode) has std 0. Guarding here turns it into
    # a zero signal rather than NaNs that would poison the whole training batch.
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std

    if clip_sigma:
        x = np.clip(x, -float(clip_sigma), float(clip_sigma))
    return np.ascontiguousarray(x, dtype=np.float32)


def preprocess_signal(waveform: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply the full configured preprocessing chain to one record.

    Args:
        waveform: Shape ``(n_samples, n_leads)`` as returned by ``wfdb.rdsamp``,
            or ``(n_leads, n_samples)``. Orientation is detected automatically.
        cfg: ECG configuration.

    Returns:
        Shape ``(n_leads, n_samples)``, ``float32``, ready for the model.
    """
    x = np.asarray(waveform, dtype=np.float32)
    n_leads = int(cfg.dataset.n_leads)

    # wfdb returns (samples, leads); the model wants (leads, samples).
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D waveform, got shape {x.shape}.")
    if x.shape[0] != n_leads and x.shape[1] == n_leads:
        x = x.T

    # A disconnected lead can arrive as NaN. Zero-fill before filtering, since
    # filtfilt propagates a single NaN across the entire record.
    if np.isnan(x).any():
        logger.debug("NaNs in waveform; zero-filling %d samples.", int(np.isnan(x).sum()))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    pre = cfg.preprocessing
    if bool(pre.get("remove_baseline_wander", True)):
        x = remove_baseline_wander(
            x,
            sampling_rate=int(cfg.dataset.sampling_rate),
            cutoff_hz=float(pre.get("highpass_hz", 0.5)),
            order=int(pre.get("filter_order", 3)),
        )

    lowpass = pre.get("lowpass_hz")
    if lowpass:
        nyquist = 0.5 * int(cfg.dataset.sampling_rate)
        sos = scipy_signal.butter(int(pre.get("filter_order", 3)),
                                  float(lowpass) / nyquist, btype="lowpass", output="sos")
        x = np.ascontiguousarray(scipy_signal.sosfiltfilt(sos, x, axis=-1), dtype=np.float32)

    x = normalize_signal(
        x,
        method=str(pre.get("normalization", "per_lead_zscore")),
        clip_sigma=pre.get("clip_sigma", 8.0),
    )

    expected_length = int(cfg.dataset.signal_length)
    if x.shape[-1] != expected_length:
        x = _fix_length(x, expected_length)
    return x


def _fix_length(x: np.ndarray, target: int) -> np.ndarray:
    """Centre-crop or zero-pad to a fixed length.

    PTB-XL records are uniformly 10 s, so this should never fire on the real
    dataset. It exists so that a truncated or corrupt file produces a usable
    tensor and a warning rather than a shape error deep inside the training loop.
    """
    current = x.shape[-1]
    if current > target:
        start = (current - target) // 2
        return x[..., start:start + target]
    pad = target - current
    left = pad // 2
    logger.warning("Record is %d samples, expected %d; zero-padding.", current, target)
    return np.pad(x, ((0, 0), (left, pad - left)), mode="constant")


def load_raw_record(record_path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one WFDB record.

    Args:
        record_path: Path **without** the ``.dat``/``.hea`` extension, as stored
            in the ``filename_lr`` column.

    Returns:
        ``(signal, header)`` with signal shape ``(n_samples, n_leads)``.
    """
    import wfdb

    signal, header = wfdb.rdsamp(str(record_path))
    return np.asarray(signal, dtype=np.float32), dict(header)


def build_waveform_cache(
    database: pd.DataFrame,
    cfg: Config,
    force: bool = False,
    root: Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Read every waveform once, preprocess it, and cache the result as ``.npy``.

    Reading ~21,800 individual WFDB files takes several minutes and is
    I/O-bound — on Colab with Drive-mounted data it is far slower than that. Doing
    it once and memory-mapping a single array afterwards makes each subsequent
    epoch read from a contiguous file, and makes a runtime restart cheap.

    At 100 Hz the cache is about 1.0 GB (``21800 x 12 x 1000 x 4`` bytes), which
    memory-maps comfortably inside a Colab session.

    Preprocessing is baked into the cache. This is safe **only** because every
    step is per-record: no statistic is shared across records, so there is nothing
    to leak between splits. If you ever add a dataset-wide scaler, it must move out
    of the cache and into a train-fitted transform.

    Args:
        database: The filtered database, in the row order the labels follow.
        cfg: ECG configuration.
        force: Rebuild even if a valid cache exists.
        root: Override the PTB-XL root.

    Returns:
        ``(waveforms, cache_path)`` with shape ``(n_records, n_leads, signal_length)``.
    """
    from tqdm.auto import tqdm

    root = root or resolve_ptbxl_root(cfg)
    sampling_rate = int(cfg.dataset.sampling_rate)
    n_leads = int(cfg.dataset.n_leads)
    length = int(cfg.dataset.signal_length)

    cache_dir = _resolve_cache_dir(cfg)
    cache_path = cache_dir / f"waveforms_{sampling_rate}hz_{len(database)}.npy"
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and not force:
        waveforms = np.load(cache_path, mmap_mode="r")
        if waveforms.shape == (len(database), n_leads, length):
            logger.info("Using cached waveforms: %s %s", cache_path.name, waveforms.shape)
            return waveforms, cache_path
        logger.warning("Cache shape %s does not match expected %s; rebuilding.",
                       waveforms.shape, (len(database), n_leads, length))

    filename_column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if filename_column not in database.columns:
        raise KeyError(
            f"{filename_column} missing from the database. PTB-XL provides filename_lr "
            "(100 Hz) and filename_hr (500 Hz); check dataset.sampling_rate."
        )

    logger.info("Building waveform cache for %d records at %d Hz (one-time, several minutes)",
                len(database), sampling_rate)

    waveforms = np.zeros((len(database), n_leads, length), dtype=np.float32)
    failures: list[dict[str, Any]] = []

    for row, (ecg_id, relative) in enumerate(
        tqdm(database[filename_column].items(), total=len(database), desc="reading ECGs")
    ):
        try:
            raw, _header = load_raw_record(root / str(relative))
            waveforms[row] = preprocess_signal(raw, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the build
            failures.append({"ecg_id": int(ecg_id), "path": str(relative), "error": str(exc)})
            logger.warning("Failed to read record %s (%s); left as zeros.", ecg_id, exc)

    np.save(cache_path, waveforms)
    save_json({
        "sampling_rate": sampling_rate,
        "n_records": int(len(database)),
        "shape": list(waveforms.shape),
        "preprocessing": cfg.preprocessing.to_dict(),
        "failures": failures,
        "size_mb": round(cache_path.stat().st_size / 1024**2, 1),
    }, meta_path)

    if failures:
        logger.warning("%d records could not be read; see %s", len(failures), meta_path.name)

    logger.info("Cache written: %s (%.1f MB)", cache_path.name,
                cache_path.stat().st_size / 1024**2)
    return np.load(cache_path, mmap_mode="r"), cache_path


def load_waveform_cache(cache_path: Path | str) -> np.ndarray:
    """Memory-map an existing waveform cache."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Waveform cache not found: {path}")
    return np.load(path, mmap_mode="r")
