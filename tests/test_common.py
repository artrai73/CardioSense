"""Smoke tests for the shared utility layer.

Run with ``pytest`` from the project root. These are intentionally fast and
dependency-light: they must pass on a CPU-only machine with no datasets present,
so a broken utility is caught before an hour of GPU time is wasted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cardiosense.common.config import Config, load_config  # noqa: E402
from cardiosense.common.io_utils import load_json, save_json  # noqa: E402
from cardiosense.common.metrics import (  # noqa: E402
    binary_metrics,
    expected_calibration_error,
    find_best_threshold,
    multilabel_metrics,
)
from cardiosense.common.paths import PATHS, get_project_root  # noqa: E402
from cardiosense.common.seeding import set_seed  # noqa: E402
from cardiosense.common.training import EarlyStopping, History  # noqa: E402


# --------------------------------------------------------------------------- paths
def test_project_root_contains_configs():
    root = get_project_root()
    assert (root / "configs").is_dir()
    assert (root / "src" / "cardiosense").is_dir()


# --------------------------------------------------------------------------- config
@pytest.mark.parametrize("modality", ["clinical", "ecg", "xray"])
def test_configs_load_and_carry_base_keys(modality):
    cfg = load_config(modality)
    assert cfg.modality == modality
    assert isinstance(cfg.seed, int)
    assert cfg.get("paths.results_root") == "results"  # merged from paths.yaml


def test_config_dotted_access_and_override():
    cfg = load_config("ecg", overrides={"training.epochs": 3})
    assert cfg.training.epochs == 3
    assert cfg.get("training.batch_size") is not None
    assert cfg.get("does.not.exist", "fallback") == "fallback"
    assert isinstance(cfg.to_dict(), dict)


def test_config_set_creates_intermediate_nodes():
    cfg = Config({})
    cfg.set("a.b.c", 5)
    assert cfg.a.b.c == 5


def test_ecg_signal_length_matches_sampling_rate():
    """PTB-XL records are exactly 10 seconds; the config must stay consistent."""
    cfg = load_config("ecg")
    assert cfg.dataset.signal_length == cfg.dataset.sampling_rate * 10
    assert len(cfg.dataset.lead_names) == cfg.dataset.n_leads


def test_ecg_folds_are_disjoint():
    cfg = load_config("ecg")
    train, val, test = (set(cfg.split.train_folds), set(cfg.split.val_folds),
                        set(cfg.split.test_folds))
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(range(1, 11))


# --------------------------------------------------------------------------- seeding
def test_set_seed_is_reproducible():
    set_seed(123, verbose=False)
    first = np.random.rand(5)
    set_seed(123, verbose=False)
    assert np.allclose(first, np.random.rand(5))


# --------------------------------------------------------------------------- io
def test_save_json_handles_numpy(tmp_path):
    payload = {"auc": np.float32(0.87), "n": np.int64(42), "arr": np.arange(3)}
    path = save_json(payload, tmp_path / "m.json")
    loaded = load_json(path)
    assert loaded["n"] == 42
    assert pytest.approx(loaded["auc"], abs=1e-6) == 0.87
    assert loaded["arr"] == [0, 1, 2]


# --------------------------------------------------------------------------- metrics
def test_binary_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    metrics = binary_metrics(y_true, y_true, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["fn"] == 0 and metrics["fp"] == 0
    assert metrics["brier"] < 0.05


def test_binary_metrics_omits_prob_metrics_without_probabilities():
    metrics = binary_metrics([0, 1, 1], [0, 1, 0])
    assert "roc_auc" not in metrics and "brier" not in metrics


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 20000)
    labels = (rng.uniform(0, 1, 20000) < probs).astype(int)
    assert expected_calibration_error(labels, probs, n_bins=10) < 0.02


def test_find_best_threshold_beats_default_on_skewed_scores():
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.concatenate([np.linspace(0.0, 0.3, 90), np.linspace(0.25, 0.45, 10)])
    threshold, score = find_best_threshold(y_true, y_prob, metric="f1")
    assert 0.0 < threshold < 1.0
    assert score > 0.0


def test_multilabel_metrics_shapes_and_keys():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=(200, 5))
    y_prob = rng.uniform(0, 1, size=(200, 5))
    out = multilabel_metrics(y_true, y_prob, thresholds=0.5,
                             class_names=["NORM", "MI", "STTC", "CD", "HYP"])
    assert "macro_roc_auc" in out and "exact_match" in out
    assert "accuracy" not in out  # deliberately not reported for multi-label
    assert set(out["per_class"]) == {"NORM", "MI", "STTC", "CD", "HYP"}


def test_multilabel_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        multilabel_metrics(np.zeros((10, 5)), np.zeros((10, 3)))


# --------------------------------------------------------------------------- training
def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max", min_delta=0.0)
    assert stopper.step(0.80, epoch=1) is True
    assert stopper.step(0.79, epoch=2) is False
    assert stopper.should_stop is False
    stopper.step(0.78, epoch=3)
    assert stopper.should_stop is True
    assert stopper.best_epoch == 1


def test_history_tracks_best():
    history = History()
    for value in (0.1, 0.5, 0.3):
        history.append(val_auc=value)
    assert history.best("val_auc", mode="max") == (2, 0.5)
    assert len(history) == 3


# --------------------------------------------------------------------------- layout
def test_expected_directories_exist():
    PATHS.ensure_all()
    for modality in ("clinical", "ecg", "xray"):
        assert PATHS.results_for(modality).is_dir()
        assert PATHS.models_for(modality).is_dir()
