"""Reading the ECG checkpoints that ``04_ecg_feature_fusion.ipynb`` actually wrote.

That notebook saves ``joint_seed{seed}.pt`` once per epoch, so the gated
joint network **is** on disk and survives a runtime disconnect. What it does
*not* save is the XGBoost booster, the fitted imputer, the scaler or the
per-class thresholds. This module recovers everything the checkpoint does
contain; :mod:`cardiosense.fusion.adapters` turns it into a prediction.

The architecture below is copied **verbatim** from the notebook. It has to be:
``load_state_dict`` matches on parameter names and shapes, so a cosmetic
rewrite -- renaming ``sig_proj``, reordering a ``Sequential``, changing a
default width -- fails to load, or worse, loads partially with ``strict=False``
and silently returns predictions from half-initialised weights.

Two subtleties in the checkpoint format are worth knowing about:

**The ``model`` key is not the best model.** ``train_joint`` saves at the end of
every epoch, so ``ckpt["model"]`` holds the *last* epoch's weights, which early
stopping had already rejected. The weights to use are
``ckpt["state"]["best_state"]``. Loading the wrong one costs accuracy with no
error message anywhere, which is why :func:`load_joint_model` prefers
``best_state`` and records which key it used.

**The feature branch cannot run from the checkpoint alone.** ``state`` carries
``feature_names`` and ``n_feat``, so the column *order* is recoverable, but the
imputer's medians and the scaler's means are not in the file. The signal branch
has no such dependency: ``signal_logits`` touches only the encoder and its own
head, so waveform-only inference works from the ``.pt`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: PTB-XL diagnostic superclasses, in the channel order the models were trained in.
ECG_CLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")

#: Matches the notebook's ``EPS`` so an ensemble reproduces its numbers exactly.
EPS = 1e-6


# --------------------------------------------------------------------------
# architecture -- verbatim from 04_ecg_feature_fusion.ipynb, do not "tidy"
# --------------------------------------------------------------------------
def _build_classes():
    """Define the modules at call time, so importing this file never needs torch."""
    import torch
    import torch.nn as nn

    class SE1d(nn.Module):
        def __init__(self, c, r=8):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                    nn.Linear(c // r, c), nn.Sigmoid())

        def forward(self, x):
            return x * self.fc(x.mean(-1)).unsqueeze(-1)

    class ResBlockSE(nn.Module):
        def __init__(self, cin, cout, stride=1, k=9, dropout=0.2):
            super().__init__()
            self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
            self.bn1 = nn.BatchNorm1d(cout)
            self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
            self.bn2 = nn.BatchNorm1d(cout)
            self.se = SE1d(cout)
            self.drop = nn.Dropout(dropout)
            self.short = (nn.Identity() if stride == 1 and cin == cout else
                          nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                        nn.BatchNorm1d(cout)))

        def forward(self, x):
            out = self.drop(torch.relu(self.bn1(self.conv1(x))))
            out = self.se(self.bn2(self.conv2(out)))
            return torch.relu(out + self.short(x))

    class SignalEncoder(nn.Module):
        """SE-ResNet-1D: (B, 12, 1000) -> (B, 2 * widths[-1]) via global avg+max pool."""

        def __init__(self, in_ch=12, widths=(64, 128, 192, 256), blocks=2, k=9, dropout=0.2):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, widths[0], 15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True))
            layers, prev = [], widths[0]
            for s, w in enumerate(widths):
                for b in range(blocks):
                    layers.append(ResBlockSE(prev, w, stride=2 if (b == 0 and s > 0) else 1,
                                             k=k, dropout=dropout))
                    prev = w
            self.stages = nn.Sequential(*layers)
            self.out_dim = prev * 2

        def forward(self, x):
            x = self.stages(self.stem(x))
            return torch.cat([x.mean(-1), x.amax(-1)], dim=1)

    class GatedJointNet(nn.Module):
        def __init__(self, n_feat, n_classes, d=256, p_moddrop=0.15):
            super().__init__()
            self.p_moddrop = p_moddrop
            self.sig_enc = SignalEncoder()
            self.sig_proj = nn.Sequential(nn.Linear(self.sig_enc.out_dim, d),
                                          nn.BatchNorm1d(d), nn.SiLU())
            self.feat_enc = nn.Sequential(
                nn.Linear(n_feat, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(512, d), nn.BatchNorm1d(d), nn.SiLU())
            self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * d, d), nn.SiLU(),
                                      nn.Dropout(0.2), nn.Linear(d, n_classes))
            self.head_sig = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_classes))
            self.head_feat = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_classes))

        def encode_signal(self, w):
            return self.sig_proj(self.sig_enc(w))

        def signal_logits(self, w):
            return self.head_sig(self.encode_signal(w))

        def forward(self, w, f):
            zs, zf = self.encode_signal(w), self.feat_enc(f)
            zs_in, zf_in = zs, zf
            if self.training and self.p_moddrop > 0:
                r = torch.rand(zs.shape[0], 1, device=zs.device)
                zs_in = zs.masked_fill(r < self.p_moddrop, 0.0)
                zf_in = zf.masked_fill((r >= self.p_moddrop) & (r < 2 * self.p_moddrop), 0.0)
            g = self.gate(torch.cat([zs_in, zf_in], dim=1))
            fused = g * zs_in + (1 - g) * zf_in
            logits = self.head(torch.cat([zs_in, zf_in, fused], dim=1))
            return logits, self.head_sig(zs), self.head_feat(zf), g

    return SE1d, ResBlockSE, SignalEncoder, GatedJointNet


def build_joint_net(n_feat: int, n_classes: int = 5, **kwargs):
    """Construct an untrained ``GatedJointNet`` with the notebook's defaults."""
    _, _, _, GatedJointNet = _build_classes()
    return GatedJointNet(n_feat, n_classes, **kwargs)


# --------------------------------------------------------------------------
# checkpoint discovery and loading
# --------------------------------------------------------------------------
#: Where ``04_ecg_feature_fusion`` writes, and where ``02_ecg_training`` wrote.
#: Searched in order; the first directory holding at least one
#: ``joint_seed*.pt`` wins.
#:
#: Both Drive-relative and models-relative spellings are listed on purpose. The
#: checkpoints land in ``models/ecg_fusion`` while every other ECG artifact
#: lives in ``models/ecg``, so depending on which of those a caller passes, the
#: target is a child, a sibling or two levels up. Listing all three spellings
#: means discovery works whichever is handed in, rather than depending on a
#: parent walk that happens to reach far enough.
CHECKPOINT_DIRS: tuple[str, ...] = (
    # relative to a Drive root, e.g. /content/drive/MyDrive/CardioSense
    "models/ecg_fusion",
    "models/ecg_fusion/no_demographics",
    "models/ecg",
    "models/ecg/checkpoints",
    # relative to a models root, e.g. .../CardioSense/models
    "ecg_fusion",
    "ecg_fusion/no_demographics",
    "ecg/checkpoints",
    # relative to models/ecg itself
    "checkpoints",
)


def find_joint_checkpoints(search_roots: Sequence[Path | str],
                           pattern: str = "joint_seed*.pt") -> list[Path]:
    """Locate per-seed joint checkpoints under any of ``search_roots``.

    Each root is tried both directly and with every entry of
    :data:`CHECKPOINT_DIRS` appended, so either a Drive root
    (``/content/drive/MyDrive/CardioSense``) or the checkpoint directory itself
    can be passed.
    """
    for root in search_roots:
        root = Path(root)
        candidates = [root] + [root / sub for sub in CHECKPOINT_DIRS]
        for directory in candidates:
            if not directory.is_dir():
                continue
            found = sorted(directory.glob(pattern))
            if found:
                return found
    return []


@dataclass
class JointCheckpoint:
    """One loaded seed, with the provenance needed to report it honestly."""

    path: Path
    n_feat: int
    feature_names: list[str]
    best_val_auc: float
    epoch: int
    finished: bool
    weights_key: str
    model: Any = field(repr=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_feat": self.n_feat,
            "n_feature_names": len(self.feature_names),
            "best_val_macro_auc": round(float(self.best_val_auc), 4),
            "epochs_trained": self.epoch,
            "early_stopped_or_finished": self.finished,
            "weights_key": self.weights_key,
        }


def _torch_load(path: Path, device: str = "cpu") -> dict[str, Any]:
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                       # torch < 2.0 has no weights_only
        return torch.load(path, map_location=device)


def _read_payload(path: Path, device: str) -> dict[str, Any]:
    """Load the raw checkpoint dict and confirm it is the layout we expect."""
    payload = _torch_load(path, device)
    if not isinstance(payload, dict) or "state" not in payload:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise KeyError(
            f"{path.name} has keys {keys}; expected the "
            "{'model', 'opt', 'sched', 'scaler', 'state'} layout written by "
            "train_joint() in 04_ecg_feature_fusion.ipynb")
    return payload


def _build_from_payload(path: Path, payload: dict[str, Any], device: str,
                        prefer_best: bool) -> JointCheckpoint:
    state = payload["state"]
    n_feat = int(state["n_feat"])

    weights, weights_key = None, ""
    if prefer_best and state.get("best_state") is not None:
        weights, weights_key = state["best_state"], "state.best_state"
    elif payload.get("model") is not None:
        weights, weights_key = payload["model"], "model"
    if weights is None:
        raise KeyError(f"{path.name} contains no usable weights")

    model = build_joint_net(n_feat, n_classes=len(ECG_CLASSES))
    model.load_state_dict(weights)          # strict: a shape change must fail loudly
    model.to(device).eval()

    return JointCheckpoint(
        path=path, n_feat=n_feat,
        feature_names=list(state.get("feature_names") or []),
        best_val_auc=float(state.get("best_auc", float("nan"))),
        epoch=int(state.get("epoch", 0)), finished=bool(state.get("done", False)),
        weights_key=weights_key, model=model,
    )


def load_joint_checkpoint(path: Path | str, device: str = "cpu",
                          prefer_best: bool = True) -> JointCheckpoint:
    """Load one ``joint_seed*.pt`` into an eval-mode model.

    Args:
        path: The checkpoint file.
        device: Where to map the tensors.
        prefer_best: Use ``state["best_state"]`` rather than ``["model"]``.
            Leave this True. ``["model"]`` is whatever the last epoch produced,
            including the epochs early stopping discarded.

    Raises:
        KeyError: If the file is not in the format ``train_joint`` writes. A
            clear failure here beats a model built on guessed keys.
        RuntimeError: If the weights do not fit the architecture. Loading with
            ``strict=False`` instead would half-initialise the model and return
            predictions with no error anywhere.
    """
    path = Path(path)
    payload = _read_payload(path, device)
    return _build_from_payload(path, payload, device, prefer_best)


def load_joint_ensemble(paths: Sequence[Path | str], device: str = "cpu",
                        prefer_best: bool = True) -> list[JointCheckpoint]:
    """Load every seed that belongs to the same run, skipping the strays.

    A checkpoint trained with a different ``USE_DEMOGRAPHICS`` setting has a
    different ``n_feat`` and a differently shaped first feature layer. Averaging
    it in would mean averaging two different models, so it is dropped with a
    warning rather than coerced.

    The feature count is read from ``state`` **before** the model is built. An
    earlier version checked it afterwards, which never ran: ``load_state_dict``
    had already raised on the shape mismatch, so a single stale file from an
    old run took the whole ensemble down instead of being skipped.
    """
    import warnings

    payloads: list[tuple[Path, dict[str, Any], int]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = _read_payload(path, device)
        payloads.append((path, payload, int(payload["state"]["n_feat"])))

    if not payloads:
        return []

    # The first checkpoint's feature count defines the run; glob() returns
    # sorted names, so this is the lowest seed number rather than an arbitrary one.
    expected = payloads[0][2]
    loaded: list[JointCheckpoint] = []
    for path, payload, n_feat in payloads:
        if n_feat != expected:
            warnings.warn(
                f"skipping {path.name}: {n_feat} features, but this ensemble is "
                f"{expected}. These are two different models -- most often one "
                "run with USE_DEMOGRAPHICS=True and one with False. Point "
                "find_joint_checkpoints at a single run's directory.",
                RuntimeWarning, stacklevel=2)
            continue
        loaded.append(_build_from_payload(path, payload, device, prefer_best))
    return loaded


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def mean_logit(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Average seeds in logit space, exactly as the notebook's ``mean_logit``.

    Averaging probabilities instead would pull an ensemble toward 0.5 and change
    every threshold the notebook tuned.
    """
    stacked = np.mean([_to_logit(p) for p in probabilities], axis=0)
    return 1.0 / (1.0 + np.exp(-stacked))


def _as_waveform_tensor(waveform: Any):
    """Coerce to a float32 ``(1, 12, 1000)`` tensor, or say why it cannot be."""
    import torch

    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"waveform must be 2-D or 3-D, got shape {array.shape}")
    if array.shape[1] != 12:
        # A (1000, 12) array is the same data transposed; accept it, because the
        # alternative is 12 channels of 1000 nonsense values and no error.
        if array.shape[2] == 12:
            array = array.transpose(0, 2, 1)
        else:
            raise ValueError(
                f"expected 12 ECG leads, got shape {array.shape}; PTB-XL at "
                "100 Hz for 10 s is (12, 1000)")
    return torch.from_numpy(np.ascontiguousarray(array))


def predict_signal_only(checkpoints: Sequence[JointCheckpoint],
                        waveform: Any) -> np.ndarray:
    """Five superclass probabilities from the waveform alone.

    Uses ``signal_logits``, which runs the encoder and its auxiliary head and
    never touches the feature branch -- so this needs no imputer and no scaler,
    only the ``.pt`` files.
    """
    import torch

    tensor = _as_waveform_tensor(waveform)
    per_seed = []
    for checkpoint in checkpoints:
        with torch.no_grad():
            logits = checkpoint.model.signal_logits(tensor)
            per_seed.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
    return mean_logit(per_seed) if len(per_seed) > 1 else np.asarray(per_seed[0])


def predict_signal_batch(checkpoints: Sequence[JointCheckpoint], waveforms: Any,
                         batch_size: int = 512) -> np.ndarray:
    """Signal-only probabilities for a whole split: ``(N, 12, 1000) -> (N, 5)``.

    Needed to re-derive per-class thresholds, which the checkpoints do not
    contain -- tuning them requires predictions over the entire validation fold,
    not one record at a time.
    """
    import torch

    array = np.asarray(waveforms, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected (N, 12, 1000), got {array.shape}")
    if array.shape[1] != 12 and array.shape[2] == 12:
        array = array.transpose(0, 2, 1)

    per_seed = []
    for checkpoint in checkpoints:
        chunks = []
        for start in range(0, len(array), batch_size):
            tensor = torch.from_numpy(
                np.ascontiguousarray(array[start:start + batch_size]))
            with torch.no_grad():
                logits = checkpoint.model.signal_logits(tensor)
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
        per_seed.append(np.concatenate(chunks, axis=0))
    return mean_logit(per_seed) if len(per_seed) > 1 else per_seed[0]


def predict_joint_batch(checkpoints: Sequence[JointCheckpoint], waveforms: Any,
                        features: Any, batch_size: int = 512) -> np.ndarray:
    """Gated-fusion probabilities for a whole split. ``features`` already scaled."""
    import torch

    waves = np.asarray(waveforms, dtype=np.float32)
    feats = np.asarray(features, dtype=np.float32)
    if len(waves) != len(feats):
        raise ValueError(
            f"{len(waves)} waveforms but {len(feats)} feature rows -- these must "
            "be the same records in the same order, or every prediction pairs "
            "one patient's ECG with another's features")

    per_seed = []
    for checkpoint in checkpoints:
        chunks = []
        for start in range(0, len(waves), batch_size):
            w = torch.from_numpy(np.ascontiguousarray(waves[start:start + batch_size]))
            f = torch.from_numpy(np.ascontiguousarray(feats[start:start + batch_size]))
            with torch.no_grad():
                logits, _, _, _ = checkpoint.model(w, f)
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
        per_seed.append(np.concatenate(chunks, axis=0))
    return mean_logit(per_seed) if len(per_seed) > 1 else per_seed[0]


def predict_joint(checkpoints: Sequence[JointCheckpoint], waveform: Any,
                  features: np.ndarray) -> tuple[np.ndarray, float]:
    """Gated fusion of waveform and features. Returns (probabilities, mean gate).

    ``features`` must already be imputed, scaled and clipped the way training
    did it -- pass the output of
    :meth:`cardiosense.fusion.adapters.ECGJointAdapter.prepare_features`, not
    raw extractor output.
    """
    import torch

    tensor = _as_waveform_tensor(waveform)
    feature_tensor = torch.as_tensor(
        np.asarray(features, dtype=np.float32).reshape(1, -1))

    expected = checkpoints[0].n_feat
    if feature_tensor.shape[1] != expected:
        raise ValueError(
            f"checkpoint expects {expected} features, got {feature_tensor.shape[1]}. "
            "This is usually a USE_DEMOGRAPHICS mismatch between the run that "
            "trained the checkpoint and the run that built the preprocessor.")

    per_seed, gates = [], []
    for checkpoint in checkpoints:
        with torch.no_grad():
            logits, _, _, gate = checkpoint.model(tensor, feature_tensor)
            per_seed.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
            gates.append(float(gate.float().mean().item()))
    probabilities = mean_logit(per_seed) if len(per_seed) > 1 else np.asarray(per_seed[0])
    return probabilities, float(np.mean(gates))


__all__ = [
    "ECG_CLASSES", "CHECKPOINT_DIRS", "JointCheckpoint",
    "build_joint_net", "find_joint_checkpoints", "load_joint_checkpoint",
    "load_joint_ensemble", "mean_logit", "predict_signal_only", "predict_joint",
    "predict_signal_batch", "predict_joint_batch",
]
