"""Tests for the ECG pipeline.

CPU-only, no network, no PTB-XL download. The fixtures build a mini dataset with
PTB-XL's exact structure — real WFDB files, stringified ``scp_codes``, nested
``records100/`` folders, a ``strat_fold`` column — so the tests exercise the real
loading path rather than a mock.

The tests that matter most:

* ``test_split_is_patient_disjoint`` — record-level splitting of PTB-XL is the
  classic leak, because patients contribute several ECGs.
* ``test_likelihood_zero_counts_as_present`` — treating ``0.0`` as "absent" would
  silently delete a large share of the positive labels.
* ``test_ig_satisfies_completeness`` — an attribution map that does not sum to
  ``F(x) - F(baseline)`` is not Integrated Gradients, whatever it is called.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cardiosense.common.config import load_config  # noqa: E402
from cardiosense.ecg.baseline import extract_features, feature_names  # noqa: E402
from cardiosense.ecg.data import (  # noqa: E402
    SUPERCLASSES,
    build_superclass_labels,
    load_metadata,
    parse_scp_codes,
    split_by_fold,
    verify_dataset,
)
from cardiosense.ecg.dataset import ECGDataset, compute_pos_weight  # noqa: E402
from cardiosense.ecg.evaluate import recommend_model, tune_per_class_thresholds  # noqa: E402
from cardiosense.ecg.explain import integrated_gradients  # noqa: E402
from cardiosense.ecg.models import build_model, model_summary  # noqa: E402
from cardiosense.ecg.preprocessing import (  # noqa: E402
    normalize_signal,
    preprocess_signal,
    remove_baseline_wander,
)

SEED = 42
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@pytest.fixture(scope="module")
def cfg():
    return load_config("ecg")


@pytest.fixture(scope="module")
def mini_ptbxl(tmp_path_factory) -> Path:
    """Write a structurally identical mini PTB-XL with real WFDB files."""
    wfdb = pytest.importorskip("wfdb")
    root = tmp_path_factory.mktemp("ptbxl")
    (root / "records100").mkdir()

    scp = {
        "NORM": ("NORM", 1), "SR": (None, 0), "IMI": ("MI", 1), "ASMI": ("MI", 1),
        "NDT": ("STTC", 1), "IRBBB": ("CD", 1), "LVH": ("HYP", 1),
    }
    pd.DataFrame([
        {"Unnamed: 0": code, "diagnostic": diag, "form": 0, "rhythm": 0,
         "diagnostic_class": cls if cls else np.nan,
         "diagnostic_subclass": cls if cls else np.nan}
        for code, (cls, diag) in scp.items()
    ]).set_index("Unnamed: 0").to_csv(root / "scp_statements.csv")

    rng = np.random.default_rng(SEED)
    rows = []
    code_cycle = ["NORM", "IMI", "NDT", "IRBBB", "LVH", "ASMI"]
    for ecg_id in range(1, 61):
        patient_id = (ecg_id - 1) % 20 + 1        # 3 records per patient
        fold = int((patient_id % 10) + 1)         # patient-consistent fold
        codes = {code_cycle[ecg_id % len(code_cycle)]: 100.0, "SR": 0.0}
        if ecg_id % 7 == 0:
            codes["NDT"] = 0.0                     # likelihood not stated -> present

        signal = rng.standard_normal((1000, 12)) * 0.3
        (root / "records100" / "00000").mkdir(exist_ok=True)
        name = f"{ecg_id:05d}_lr"
        wfdb.wrsamp(record_name=name, fs=100, units=["mV"] * 12, sig_name=LEADS,
                    p_signal=signal, write_dir=str(root / "records100" / "00000"),
                    fmt=["16"] * 12)
        rows.append({
            "ecg_id": ecg_id, "patient_id": patient_id, "age": 50, "sex": ecg_id % 2,
            "scp_codes": str(codes), "strat_fold": fold,
            "filename_lr": f"records100/00000/{name}",
            "filename_hr": f"records500/00000/{ecg_id:05d}_hr",
        })
    pd.DataFrame(rows).set_index("ecg_id").to_csv(root / "ptbxl_database.csv")
    return root


@pytest.fixture(scope="module")
def labelled(mini_ptbxl, cfg):
    local = load_config("ecg", overrides={"dataset.root": str(mini_ptbxl)})
    database, statements = load_metadata(local, mini_ptbxl)
    return (*build_superclass_labels(database, statements, local), local)


# --------------------------------------------------------------------- config
def test_class_order_is_fixed(cfg):
    """Class order defines every matrix column; it must not drift."""
    assert tuple(cfg.task.classes) == SUPERCLASSES
    assert cfg.task.type == "multilabel"


def test_verify_dataset_reports_a_helpful_error(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="physionet"):
        verify_dataset(cfg, root=tmp_path / "does_not_exist")


def test_verify_dataset_accepts_the_mini_dataset(mini_ptbxl, cfg):
    report = verify_dataset(cfg, root=mini_ptbxl)
    assert report["record_folders"] >= 1


# ---------------------------------------------------------------------- labels
def test_parse_scp_codes_handles_the_stringified_dict():
    parsed = parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}")
    assert parsed == {"NORM": 100.0, "SR": 0.0}


def test_parse_scp_codes_survives_garbage():
    assert parse_scp_codes("not a dict") == {}
    assert parse_scp_codes(np.nan) == {}


def test_labels_are_multi_hot_with_correct_width(labelled):
    _database, labels, _report, local = labelled
    assert labels.shape[1] == len(local.task.classes)
    assert set(np.unique(labels)) <= {0.0, 1.0}


def test_non_diagnostic_codes_are_ignored(labelled):
    """`SR` is a rhythm statement, not diagnostic; it must not create a label."""
    _database, labels, report, _local = labelled
    assert "SR" not in report["unmapped_diagnostic_codes"]
    assert labels.sum() > 0


def test_likelihood_zero_counts_as_present(mini_ptbxl):
    """A likelihood of 0.0 means 'not stated', NOT 'absent'.

    Reading it as absent would silently drop a large share of PTB-XL's positives.
    """
    local = load_config("ecg", overrides={"dataset.root": str(mini_ptbxl)})
    database, statements = load_metadata(local, mini_ptbxl)
    _db, labels, _report = build_superclass_labels(database, statements, local)

    sttc_index = list(local.task.classes).index("STTC")
    zero_likelihood_rows = [
        row for row, codes in enumerate(database.scp_codes)
        if parse_scp_codes(codes).get("NDT") == 0.0
    ]
    assert zero_likelihood_rows, "fixture should contain a 0.0-likelihood NDT"
    for row in zero_likelihood_rows:
        assert labels[row, sttc_index] == 1.0


# ---------------------------------------------------------------------- splits
def test_split_is_patient_disjoint(labelled):
    """The whole point of using the official strat_fold column."""
    database, labels, _report, local = labelled
    splits = split_by_fold(database, labels, local)
    assert sum(splits["patient_overlap"].values()) == 0

    patients = {name: set(splits[name]["database"].patient_id)
                for name in ("train", "val", "test")}
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["val"].isdisjoint(patients["test"])


def test_split_rejects_overlapping_folds(labelled):
    database, labels, _report, local = labelled
    bad = load_config("ecg", overrides={
        "dataset.root": str(local.dataset.root),
        "split.train_folds": [1, 2, 3],
        "split.val_folds": [3],          # 3 appears twice
        "split.test_folds": [10],
    })
    with pytest.raises(ValueError, match="overlap"):
        split_by_fold(database, labels, bad)


def test_split_covers_every_record(labelled):
    database, labels, _report, local = labelled
    splits = split_by_fold(database, labels, local)
    total = sum(len(splits[name]["indices"]) for name in ("train", "val", "test"))
    assert total == len(database)


# --------------------------------------------------------------- preprocessing
def test_highpass_removes_baseline_drift():
    """A 0.15 Hz drift should be strongly attenuated; the QRS band should not be."""
    fs, n = 100, 1000
    t = np.arange(n) / fs
    drift = 2.0 * np.sin(2 * np.pi * 0.15 * t)
    beat = np.sin(2 * np.pi * 10 * t)
    signal = np.vstack([drift + beat] * 12)

    filtered = remove_baseline_wander(signal, sampling_rate=fs, cutoff_hz=0.5)
    assert filtered.std() < signal.std()
    # The 10 Hz component should survive largely intact.
    spectrum = np.abs(np.fft.rfft(filtered[0]))
    frequencies = np.fft.rfftfreq(n, 1 / fs)
    assert spectrum[np.argmin(np.abs(frequencies - 10))] > 0.5 * spectrum.max()


def test_per_lead_zscore_standardises_each_lead_independently():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((12, 1000)) * np.arange(1, 13)[:, None]
    out = normalize_signal(signal, method="per_lead_zscore", clip_sigma=None)
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(out.std(axis=-1), 1, atol=1e-5)


def test_normalization_survives_a_flat_lead():
    """A disconnected electrode has std 0 and must not produce NaNs."""
    signal = np.random.default_rng(0).standard_normal((12, 1000))
    signal[3] = 0.0
    out = normalize_signal(signal, method="per_lead_zscore")
    assert not np.isnan(out).any()


def test_clipping_caps_electrode_pops():
    signal = np.random.default_rng(0).standard_normal((12, 1000))
    signal[0, 500] = 500.0
    out = normalize_signal(signal, method="per_lead_zscore", clip_sigma=8.0)
    assert np.abs(out).max() <= 8.0 + 1e-6


def test_preprocess_transposes_wfdb_orientation(cfg):
    """wfdb returns (samples, leads); the model needs (leads, samples)."""
    raw = np.random.default_rng(0).standard_normal((1000, 12))
    out = preprocess_signal(raw, cfg)
    assert out.shape == (12, 1000)
    assert out.dtype == np.float32


def test_preprocess_handles_nans(cfg):
    raw = np.random.default_rng(0).standard_normal((1000, 12))
    raw[10:20, 4] = np.nan
    assert not np.isnan(preprocess_signal(raw, cfg)).any()


# -------------------------------------------------------------------- dataset
def test_dataset_returns_writable_tensors():
    """Memory-mapped slices are read-only; handing them to torch is unsafe."""
    import tempfile

    path = Path(tempfile.mkdtemp()) / "w.npy"
    np.save(path, np.zeros((10, 12, 1000), dtype=np.float32))
    mmapped = np.load(path, mmap_mode="r")
    labels = np.zeros((10, 5), dtype=np.float32)

    dataset = ECGDataset(mmapped, labels, indices=[0, 1, 2], class_names=list(SUPERCLASSES))
    waveform, label = dataset[0]
    assert waveform.shape == (12, 1000)
    assert label.shape == (5,)
    waveform[0, 0] = 1.0        # would raise on a read-only buffer
    label[0] = 1.0


def test_dataset_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="aligned"):
        ECGDataset(np.zeros((10, 12, 100)), np.zeros((5, 5)), indices=[0])


def test_pos_weight_is_larger_for_rarer_classes():
    labels = np.zeros((100, 5), dtype=np.float32)
    labels[:50, 0] = 1      # common
    labels[:5, 1] = 1       # rare
    weights = compute_pos_weight(labels)
    assert weights[1] > weights[0]
    assert weights.shape == (5,)


# --------------------------------------------------------------------- models
@pytest.mark.parametrize("name", ["cnn1d", "resnet1d"])
def test_models_produce_logits_of_the_right_shape(cfg, name):
    model = build_model(cfg, name=name)
    out = model(torch.zeros(3, 12, 1000))
    assert out.shape == (3, 5)
    # Raw logits, not probabilities: values must be free to leave [0, 1].
    assert not ((out >= 0).all() and (out <= 1).all() and out.abs().sum() > 0)


def test_cnn_is_not_gratuitously_large(cfg):
    summary = model_summary(build_model(cfg, name="cnn1d"))
    assert summary["total"] < 5_000_000, "the CNN should stay small for ~17k records"


def test_model_handles_a_batch_of_one(cfg):
    """BatchNorm with batch size 1 fails in train mode; eval must still work."""
    model = build_model(cfg, name="cnn1d").eval()
    assert model(torch.zeros(1, 12, 1000)).shape == (1, 5)


# ------------------------------------------------------------------ evaluation
def test_thresholds_are_tuned_per_class(cfg):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, 5)).astype(float)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, (200, 5)), 0.01, 0.99)
    thresholds, info = tune_per_class_thresholds(y, p, cfg)
    assert thresholds.shape == (5,)
    assert info["tuned_on"] == "validation"
    assert len(set(np.round(thresholds, 3))) > 1, "per-class tuning should differ by class"


def test_recommendation_keeps_the_simpler_model_on_a_small_gain():
    results = {
        "cnn1d": {"metrics": {"macro_roc_auc": 0.900}, "parameters": 370_000},
        "resnet1d": {"metrics": {"macro_roc_auc": 0.905}, "parameters": 4_000_000},
    }
    decision = recommend_model(results)
    assert decision["recommended"] == "cnn1d"


def test_recommendation_upgrades_on_a_real_gain():
    results = {
        "cnn1d": {"metrics": {"macro_roc_auc": 0.870}, "parameters": 370_000},
        "resnet1d": {"metrics": {"macro_roc_auc": 0.920}, "parameters": 4_000_000},
    }
    assert recommend_model(results)["recommended"] == "resnet1d"


def test_recommendation_warns_when_the_baseline_wins():
    results = {
        "statistical_baseline": {"metrics": {"macro_roc_auc": 0.80}, "parameters": 600},
        "cnn1d": {"metrics": {"macro_roc_auc": 0.79}, "parameters": 370_000},
    }
    decision = recommend_model(results)
    assert decision["recommended"] == "statistical_baseline"
    assert "warning" in decision


# -------------------------------------------------------------------- baseline
def test_feature_extraction_shape_and_names(cfg):
    features = list(cfg.baseline.features)
    waveform = np.random.default_rng(0).standard_normal((12, 1000))
    vector = extract_features(waveform, features)
    assert vector.shape == (12 * len(features),)
    assert len(feature_names(features, LEADS)) == vector.shape[0]
    assert np.isfinite(vector).all()


# ------------------------------------------------------------- explainability
def test_ig_satisfies_completeness(cfg):
    """Attributions must sum to F(x) - F(baseline). That is what makes it IG."""
    torch.manual_seed(SEED)
    model = build_model(cfg, name="cnn1d").eval()
    inputs = torch.randn(2, 12, 1000)

    attributions, info = integrated_gradients(model, inputs, target_class=1, n_steps=128)

    assert attributions.shape == (2, 12, 1000)
    assert info["converged"], (
        f"IG failed completeness: relative error "
        f"{info['mean_relative_convergence_error']}"
    )
    assert info["mean_absolute_convergence_error"] < 0.05


def test_manual_ig_matches_captum(cfg):
    """The fallback implementation must agree with Captum, not merely run."""
    pytest.importorskip("captum")
    torch.manual_seed(SEED)
    model = build_model(cfg, name="cnn1d").eval()
    inputs = torch.randn(1, 12, 1000)

    captum_attr, _ = integrated_gradients(model, inputs, 0, n_steps=256, use_captum=True)
    manual_attr, _ = integrated_gradients(model, inputs, 0, n_steps=256, use_captum=False)

    correlation = np.corrcoef(captum_attr.ravel(), manual_attr.ravel())[0, 1]
    assert correlation > 0.99, f"implementations disagree (r = {correlation:.4f})"
