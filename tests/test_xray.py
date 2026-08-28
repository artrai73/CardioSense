"""Tests for the chest X-ray pipeline.

CPU-only, no network, no 45 GB download. The fixture builds a mini dataset with
ChestX-ray14's exact structure — flat PNG folder, pipe-separated ``Finding
Labels``, ``Patient ID`` column, NIH-style patient-disjoint list files — so the
tests exercise the real loading and splitting path.

The tests that matter most:

* ``test_split_is_patient_disjoint`` and
  ``test_image_level_split_would_leak`` — patient-level splitting is the single
  decision that most affects whether the reported AUC is real.
* ``test_horizontal_flip_is_refused`` — a silent config change here would teach
  the model that a right-sided heart is normal.
* ``test_eval_transform_is_deterministic`` — randomness on val/test makes early
  stopping lucky and results irreproducible.
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
from cardiosense.xray.baseline import extract_image_features, majority_class_baseline  # noqa: E402
from cardiosense.xray.data import (  # noqa: E402
    build_target,
    load_metadata,
    split_by_patient,
    verify_dataset,
)
from cardiosense.xray.dataset import ChestXrayDataset, compute_pos_weight  # noqa: E402
from cardiosense.xray.evaluate import (  # noqa: E402
    evaluate_binary,
    recommend_model,
    select_error_examples,
    tune_threshold,
)
from cardiosense.xray.explain import GradCAM, compute_gradcam  # noqa: E402
from cardiosense.xray.models import XrayDenseNet121, model_summary, set_backbone_trainable  # noqa: E402
from cardiosense.xray.preprocessing import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
    denormalize,
)

SEED = 42
IMAGE_SIZE = 64      # small probe size keeps the tests fast


@pytest.fixture(scope="module")
def cfg():
    return load_config("xray")


@pytest.fixture(scope="module")
def mini_nih(tmp_path_factory) -> Path:
    """Write a structurally identical mini ChestX-ray14."""
    from PIL import Image

    root = tmp_path_factory.mktemp("nih")
    (root / "images").mkdir()

    rng = np.random.default_rng(SEED)
    rows = []
    image_id = 0
    for patient_id in range(1, 41):
        has_target = patient_id % 5 == 0            # 20% of patients
        for _ in range(int(rng.integers(1, 5))):    # several films per patient
            image_id += 1
            positive = has_target and rng.random() < 0.9
            labels = ["Cardiomegaly"] if positive else []
            if rng.random() < 0.3:
                labels.append("Effusion")
            name = f"{patient_id:08d}_{image_id:03d}.png"

            array = (rng.uniform(0.2, 0.8, (96, 96)) * 255).astype(np.uint8)
            Image.fromarray(array, mode="L").save(root / "images" / name)

            rows.append({
                "Image Index": name,
                "Finding Labels": "|".join(labels) if labels else "No Finding",
                "Patient ID": patient_id,
                "Patient Age": int(rng.integers(20, 90)),
                "Patient Gender": "M" if rng.random() < 0.5 else "F",
                "View Position": "PA" if rng.random() < 0.7 else "AP",
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "Data_Entry_2017_v2020.csv", index=False)

    test_patients = set(range(1, 11))
    test_images = frame[frame["Patient ID"].isin(test_patients)]["Image Index"]
    trainval_images = frame[~frame["Patient ID"].isin(test_patients)]["Image Index"]
    (root / "test_list.txt").write_text("\n".join(test_images))
    (root / "train_val_list.txt").write_text("\n".join(trainval_images))
    return root


@pytest.fixture(scope="module")
def local_cfg(mini_nih):
    return load_config("xray", overrides={
        "dataset.root": str(mini_nih),
        "dataset.negative_ratio": None,
        "preprocessing.image_size": IMAGE_SIZE,
    })


@pytest.fixture(scope="module")
def targeted(mini_nih, local_cfg):
    frame = load_metadata(local_cfg, mini_nih)
    return build_target(frame, local_cfg)


# ------------------------------------------------------------------- dataset
def test_verify_dataset_gives_download_instructions(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        verify_dataset(cfg, root=tmp_path / "missing")


def test_verify_accepts_either_metadata_filename(mini_nih, local_cfg):
    """The Kaggle mirror and the NIH v2020 release use different filenames."""
    report = verify_dataset(local_cfg, root=mini_nih)
    assert report["n_image_files"] > 0
    assert report["has_official_lists"]

    renamed = mini_nih / "Data_Entry_2017_v2020.csv"
    alias = mini_nih / "Data_Entry_2017.csv"
    renamed.rename(alias)
    try:
        assert verify_dataset(local_cfg, root=mini_nih)["n_image_files"] > 0
    finally:
        alias.rename(renamed)


def test_target_comes_from_the_released_labels(targeted, local_cfg):
    frame, report = targeted
    assert set(frame.target.unique()) <= {0, 1}
    # Every positive must literally contain the label string. Nothing invented.
    positives = frame[frame.target == 1]
    assert positives[local_cfg.dataset.label_column].str.contains("Cardiomegaly").all()
    assert report["positives"] == int(frame.target.sum())


def test_view_filter_keeps_only_pa(targeted, local_cfg):
    """AP films magnify the cardiac silhouette; mixing views creates a shortcut."""
    frame, report = targeted
    assert report["view_filter"] == "PA"
    assert (frame[local_cfg.dataset.view_column] == "PA").all()


def test_negative_subsampling_preserves_patient_grouping(mini_nih):
    """Subsampling must drop whole patients, or the later split can leak."""
    local = load_config("xray", overrides={
        "dataset.root": str(mini_nih), "dataset.negative_ratio": 2,
    })
    frame = load_metadata(local, mini_nih)
    filtered, report = build_target(frame, local)

    assert report["negative_subsampling"]["sampled_by"] == "patient"
    # No patient may be partially represented: every image of a kept patient stays.
    original = build_target(load_metadata(local, mini_nih),
                            load_config("xray", overrides={
                                "dataset.root": str(mini_nih),
                                "dataset.negative_ratio": None}))[0]
    for patient, group in filtered.groupby("Patient ID"):
        assert len(group) == len(original[original["Patient ID"] == patient])


# --------------------------------------------------------------------- split
def test_split_is_patient_disjoint(targeted, local_cfg, mini_nih):
    frame, _report = targeted
    splits = split_by_patient(frame, local_cfg, mini_nih)

    assert sum(splits["summary"]["patient_overlap"].values()) == 0
    patients = {name: set(splits[name]["Patient ID"]) for name in ("train", "val", "test")}
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])


def test_image_level_split_would_leak(targeted):
    """Demonstrates the failure mode the patient-level split exists to prevent."""
    frame, _report = targeted
    rng = np.random.default_rng(0)
    shuffled = frame.sample(frac=1.0, random_state=0)
    half = len(shuffled) // 2
    naive_train = set(shuffled.iloc[:half]["Patient ID"])
    naive_test = set(shuffled.iloc[half:]["Patient ID"])

    assert len(naive_train & naive_test) > 0, (
        "The fixture must contain patients with multiple images, or this test "
        "cannot demonstrate the leak it is guarding against."
    )


def test_split_covers_every_image(targeted, local_cfg, mini_nih):
    frame, _report = targeted
    splits = split_by_patient(frame, local_cfg, mini_nih)
    total = sum(len(splits[name]) for name in ("train", "val", "test"))
    assert total == len(frame)


def test_split_is_reproducible(targeted, local_cfg, mini_nih):
    frame, _report = targeted
    first = split_by_patient(frame, local_cfg, mini_nih)
    second = split_by_patient(frame, local_cfg, mini_nih)
    assert list(first["test"]["Image Index"]) == list(second["test"]["Image Index"])


# ------------------------------------------------------------- preprocessing
def test_horizontal_flip_is_refused(mini_nih):
    """Mirroring a chest X-ray creates dextrocardia. The code must refuse."""
    local = load_config("xray", overrides={
        "dataset.root": str(mini_nih), "augmentation.horizontal_flip": True,
    })
    with pytest.raises(ValueError, match="dextrocardia"):
        build_train_transform(local)


def test_vertical_flip_is_refused(mini_nih):
    local = load_config("xray", overrides={
        "dataset.root": str(mini_nih), "augmentation.vertical_flip": True,
    })
    with pytest.raises(ValueError, match="upside-down"):
        build_train_transform(local)


def test_eval_transform_is_deterministic(local_cfg):
    """Randomness on val/test would make early stopping lucky and results unstable."""
    from PIL import Image

    transform = build_eval_transform(local_cfg)
    image = Image.fromarray(
        (np.random.default_rng(0).uniform(0, 1, (96, 96)) * 255).astype(np.uint8), mode="L"
    )
    first, second = transform(image), transform(image)
    assert torch.allclose(first, second)
    assert first.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_train_transform_is_stochastic(local_cfg):
    from PIL import Image

    transform = build_train_transform(local_cfg)
    image = Image.fromarray(
        (np.random.default_rng(0).uniform(0, 1, (96, 96)) * 255).astype(np.uint8), mode="L"
    )
    torch.manual_seed(0)
    first = transform(image)
    torch.manual_seed(1)
    second = transform(image)
    assert not torch.allclose(first, second), "augmentation is not actually applying"


def test_denormalize_returns_unit_range(local_cfg):
    tensor = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    out = denormalize(tensor, local_cfg)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ------------------------------------------------------------------- dataset
def test_dataset_returns_tensor_and_target(targeted, local_cfg, mini_nih):
    frame, _report = targeted
    dataset = ChestXrayDataset(frame, mini_nih / "images", build_eval_transform(local_cfg))
    image, target = dataset[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert target.shape == (1,)
    assert float(target) in {0.0, 1.0}


def test_dataset_requires_a_target_column(mini_nih, local_cfg):
    with pytest.raises(KeyError, match="target"):
        ChestXrayDataset(pd.DataFrame({"Image Index": ["a.png"]}),
                         mini_nih / "images", build_eval_transform(local_cfg))


def test_pos_weight_reflects_imbalance():
    targets = np.concatenate([np.ones(5), np.zeros(95)])
    weight = compute_pos_weight(targets)
    assert float(weight) == pytest.approx(19.0)


def test_pos_weight_is_capped():
    targets = np.concatenate([np.ones(1), np.zeros(9999)])
    assert float(compute_pos_weight(targets, cap=50.0)) == 50.0


# -------------------------------------------------------------------- models
def test_model_outputs_one_logit():
    model = XrayDenseNet121(pretrained=False, n_classes=1)
    out = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (2, 1)


def test_freeze_then_unfreeze_from_block():
    model = XrayDenseNet121(pretrained=False)

    frozen = set_backbone_trainable(model, trainable=False)
    assert frozen["trainable_blocks"] == []
    head_only = frozen["trainable_parameters"]

    unfrozen = set_backbone_trainable(model, trainable=True, from_block="denseblock3")
    assert "denseblock3" in unfrozen["trainable_blocks"]
    assert "denseblock4" in unfrozen["trainable_blocks"]
    assert "denseblock1" in unfrozen["frozen_blocks"]
    assert unfrozen["trainable_parameters"] > head_only


def test_unknown_block_raises():
    model = XrayDenseNet121(pretrained=False)
    with pytest.raises(ValueError, match="Unknown block"):
        set_backbone_trainable(model, True, from_block="denseblock9")


def test_feature_map_retains_spatial_structure():
    """Grad-CAM needs spatial feature maps; a pooled vector would be useless."""
    model = XrayDenseNet121(pretrained=False)
    summary = model_summary(model, input_shape=(1, 3, 224, 224))
    assert summary["feature_map_shape"][0] == 1
    assert summary["feature_map_shape"][1] == 1024
    assert summary["feature_map_shape"][2] == 7 and summary["feature_map_shape"][3] == 7


# ---------------------------------------------------------------- evaluation
def test_majority_baseline_exposes_the_accuracy_trap():
    y_train = np.concatenate([np.ones(5), np.zeros(95)])
    y_test = np.concatenate([np.ones(5), np.zeros(95)])
    result = majority_class_baseline(y_train, y_test)

    assert result["metrics"]["accuracy"] == pytest.approx(0.95)
    assert result["metrics"]["recall"] == 0.0        # finds nothing
    assert result["metrics"]["tp"] == 0


def test_evaluate_records_the_pr_auc_chance_level(cfg):
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(20), np.zeros(180)]).astype(int)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 200), 0.01, 0.99)

    block = evaluate_binary(y, p, 0.5, cfg, with_ci=False)
    assert block["pr_auc_chance_level"] == pytest.approx(0.1)
    assert block["accuracy_of_always_negative"] == pytest.approx(0.9)
    assert "at_default_threshold_0.5" in block


def test_threshold_is_tuned_on_validation(cfg):
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(30), np.zeros(170)]).astype(int)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 200), 0.01, 0.99)
    threshold, info = tune_threshold(y, p, cfg)
    assert 0.0 < threshold < 1.0
    assert info["tuned_on"] == "validation"


def test_error_selection_prefers_confident_mistakes():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.02, 0.45, 0.95, 0.55])   # FN at 0.02 is the interesting one
    selection = select_error_examples(y_true, y_prob, threshold=0.5, n_per_category=1)
    assert selection["FN"] == [0]                 # most confident miss first
    assert selection["FP"] == [2]                 # most confident false alarm first


def test_recommendation_warns_when_pixels_beat_the_cnn():
    results = {
        "majority_class": {"metrics": {"at_tuned_threshold": {"pr_auc": 0.10}}},
        "logreg_pixel_features": {"metrics": {"at_tuned_threshold": {"pr_auc": 0.42}}},
        "densenet121": {"metrics": {"at_tuned_threshold": {"pr_auc": 0.43}}},
    }
    decision = recommend_model(results)
    assert decision["recommended"] == "logreg_pixel_features"
    assert "warning" in decision


def test_recommendation_accepts_a_real_gain():
    results = {
        "majority_class": {"metrics": {"at_tuned_threshold": {"pr_auc": 0.10}}},
        "logreg_pixel_features": {"metrics": {"at_tuned_threshold": {"pr_auc": 0.25}}},
        "densenet121": {"metrics": {"at_tuned_threshold": {"pr_auc": 0.55}}},
    }
    assert recommend_model(results)["recommended"] == "densenet121"


# -------------------------------------------------------------------- baseline
def test_image_features_have_the_expected_width(mini_nih):
    path = next((mini_nih / "images").glob("*.png"))
    vector = extract_image_features(path, downsample_size=16, hist_bins=8)
    assert vector.shape == (16 * 16 + 8,)
    assert np.isfinite(vector).all()


# -------------------------------------------------------------------- Grad-CAM
def test_gradcam_matches_input_resolution():
    model = XrayDenseNet121(pretrained=False).eval()
    images = torch.randn(2, 3, 224, 224)
    cams, probabilities = compute_gradcam(model, images)

    assert cams.shape == (2, 224, 224)
    assert probabilities.shape == (2,)
    assert cams.min() >= 0.0 and cams.max() <= 1.0 + 1e-6


def test_gradcam_removes_its_hooks():
    """A leaked backward hook silently slows every later forward pass."""
    model = XrayDenseNet121(pretrained=False).eval()
    layer = model.features.denseblock4

    before = len(layer._forward_hooks) + len(layer._backward_hooks)
    with GradCAM(model, "features.denseblock4") as explainer:
        explainer(torch.randn(1, 3, 224, 224))
        during = len(layer._forward_hooks) + len(layer._backward_hooks)
    after = len(layer._forward_hooks) + len(layer._backward_hooks)

    assert during > before
    assert after == before


def test_gradcam_rejects_an_unknown_layer():
    model = XrayDenseNet121(pretrained=False)
    with pytest.raises(AttributeError, match="not found"):
        GradCAM(model, "features.does_not_exist")
