"""Fusion adapters against artifacts saved exactly the way Phase 1 saves them.

``test_fusion_rag.py`` exercises the fusion maths with stand-in models. These
tests build real clinical and X-ray artifacts through the Phase 1 code paths --
``save_pickle`` (joblib), ``make_prefit_calibrator``, ``XrayDenseNet121`` and the
``torch.save`` layout of ``xray/train.py`` -- and check that the adapters load
them and reproduce Phase 1's own inference. Each test pins a bug that the
stand-ins could not see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cardiosense.common.io_utils import save_pickle  # noqa: E402
from cardiosense.fusion.adapters import ClinicalAdapter, XrayAdapter  # noqa: E402
from cardiosense.fusion.schema import UNCALIBRATED_RELIABILITY, reliability_from_ece  # noqa: E402


# --------------------------------------------------------------------------
# clinical
# --------------------------------------------------------------------------
@pytest.fixture
def clinical_artifacts(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from cardiosense.common.compat import make_prefit_calibrator

    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame({
        "age": rng.integers(30, 80, n).astype(float),
        "thalach": rng.normal(150, 20, n),
        "cp": rng.integers(1, 5, n),
        "sex": rng.integers(0, 2, n),
    })
    y = ((frame.age / 80 + (frame.cp == 4) * 0.5 + rng.normal(0, 0.3, n)) > 1.0).astype(int).to_numpy()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ["age", "thalach"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["cp", "sex"]),
    ])
    X_train = preprocessor.fit_transform(frame.iloc[:200])
    model = LogisticRegression(max_iter=1000).fit(X_train, y[:200])
    # exactly how clinical/calibrate.py builds it: a wrapper around the fitted model
    calibrator = make_prefit_calibrator(model, method="sigmoid")
    calibrator.fit(preprocessor.transform(frame.iloc[200:]), y[200:])

    # exactly how clinical/train.py saves them: save_pickle -> joblib.dump
    save_pickle(model, tmp_path / "clinical_model.pkl")
    save_pickle(preprocessor, tmp_path / "clinical_preprocessor.pkl")
    save_pickle(calibrator, tmp_path / "clinical_calibrator.pkl")
    metadata = {
        "model_version": "clinical-test",
        "threshold": 0.4,
        "calibration": {"method": "sigmoid", "fitted_on": "validation split",
                        "metrics": {"uncalibrated": {"ece": 0.11}, "calibrated": {"ece": 0.05}}},
    }
    (tmp_path / "clinical_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path, frame, preprocessor, calibrator


def test_clinical_adapter_reads_joblib_artifacts(clinical_artifacts):
    """Phase 1 saves with joblib.dump; plain pickle.load cannot read that."""
    directory, *_ = clinical_artifacts
    adapter = ClinicalAdapter(directory)
    assert adapter.available, adapter.load_error


def test_clinical_calibrator_gets_the_feature_matrix(clinical_artifacts):
    """The calibrator wraps the model, so it must see features, not a probability.

    Passing it the raw probability as a one-column matrix raised a shape error
    on every patient, which the adapter turned into a failed prediction.
    """
    directory, frame, preprocessor, calibrator = clinical_artifacts
    adapter = ClinicalAdapter(directory)
    row = frame.iloc[[250]]
    prediction = adapter.predict(row.iloc[0].to_dict())

    assert prediction.available and prediction.risk_score is not None, prediction.reason
    expected = float(calibrator.predict_proba(preprocessor.transform(row))[0, 1])
    assert prediction.risk_score == pytest.approx(expected, abs=1e-9)


def test_clinical_reliability_uses_the_phase1_ece_key(clinical_artifacts):
    """train.py stores the held-out ECE at calibration.metrics.calibrated.ece."""
    directory, *_ = clinical_artifacts
    adapter = ClinicalAdapter(directory)
    assert adapter.is_calibrated
    assert adapter.reliability == pytest.approx(reliability_from_ece(0.05))
    assert adapter.reliability > UNCALIBRATED_RELIABILITY


# --------------------------------------------------------------------------
# X-ray
# --------------------------------------------------------------------------
@pytest.fixture
def xray_artifacts(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    Image = pytest.importorskip("PIL.Image")

    from cardiosense.xray.models import XrayDenseNet121

    torch.manual_seed(0)
    net = XrayDenseNet121(pretrained=False, n_classes=1, dropout=0.2).eval()
    # exactly the layout xray/train.py writes
    torch.save({"model_state": net.state_dict(), "model_name": "densenet121", "threshold": 0.45},
               tmp_path / "xray_model.pth")
    config = {
        "model_version": "xray-test", "model_name": "densenet121", "n_classes": 1, "dropout": 0.2,
        "threshold": 0.45, "image_size": 64,
        "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
        "to_rgb": True, "training_prevalence": 0.0343,
    }
    (tmp_path / "xray_config.json").write_text(json.dumps(config), encoding="utf-8")

    rng = np.random.default_rng(1)
    image_path = tmp_path / "film.png"
    # an RGBA file with colour in it, like some ChestX-ray14 PNGs
    Image.fromarray(rng.integers(0, 255, (96, 96, 4), dtype=np.uint8), mode="RGBA").save(image_path)
    return tmp_path, net, config, image_path


def _phase1_probability(net, config, image_path):
    """Phase 1's own inference path: xray/predict.py's transform on an 'L' image."""
    import torch
    from PIL import Image
    from torchvision import transforms as T

    size = int(config["image_size"])
    transform = T.Compose([
        T.Resize((size, size)), T.Grayscale(num_output_channels=3), T.ToTensor(),
        T.Normalize(mean=config["normalize_mean"], std=config["normalize_std"]),
    ])
    with Image.open(image_path) as handle:
        tensor = transform(handle.convert("L")).unsqueeze(0)
    with torch.no_grad():
        return float(torch.sigmoid(net(tensor).squeeze()).item())


def test_xray_adapter_reproduces_phase1_inference(xray_artifacts):
    """Same weights, same architecture, same preprocessing -> same probability."""
    directory, net, config, image_path = xray_artifacts
    adapter = XrayAdapter(directory)
    assert adapter.available, adapter.load_error

    prediction = adapter.predict(str(image_path))
    assert prediction.available, prediction.reason
    assert prediction.risk_score == pytest.approx(_phase1_probability(net, config, image_path), abs=1e-5)


def test_xray_mismatched_weights_fail_loudly(xray_artifacts):
    """A non-strict load that matches nothing would return a random network."""
    import torch

    directory, *_ = xray_artifacts
    torch.save({"model_state": {"not.a.real.key": torch.zeros(1)}}, directory / "xray_model.pth")
    adapter = XrayAdapter(directory)
    prediction = adapter.predict(str(directory / "film.png"))
    assert not prediction.available or prediction.risk_score is None


def test_xray_without_population_prevalence_is_not_labelled_corrected(xray_artifacts):
    """Phase 1 writes no population prevalence, so no correction may be claimed."""
    directory, _, _, image_path = xray_artifacts
    adapter = XrayAdapter(directory)
    assert not adapter.applies_prior_correction
    assert adapter.status()["prior_correction"].startswith("not applied")

    prediction = adapter.predict(str(image_path))
    assert "cardiomegaly_population_referenced" not in prediction.class_probabilities


def test_xray_population_prevalence_enables_the_correction(xray_artifacts):
    directory, _, config, image_path = xray_artifacts
    config = {**config, "population_prevalence_for_prior_correction": 0.025}
    (directory / "xray_config.json").write_text(json.dumps(config), encoding="utf-8")

    adapter = XrayAdapter(directory)
    assert adapter.applies_prior_correction
    prediction = adapter.predict(str(image_path))
    probs = prediction.class_probabilities
    assert probs["cardiomegaly_population_referenced"] < probs["cardiomegaly_cohort_referenced"]
    assert prediction.risk_score == pytest.approx(probs["cardiomegaly_population_referenced"])
