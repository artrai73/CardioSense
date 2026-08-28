"""Tests for the clinical pipeline.

CPU-only, no network, no dataset download — the fixture below builds a small
frame with the same schema as UCI Heart Disease. It is a **test fixture**, used
only to exercise code paths; it is never written into the repository and produces
no reportable numbers.

The tests that matter most are the leakage checks: ``test_preprocessor_is_fit_on_train_only``
and ``test_split_has_no_overlap``. Those are the failures that would silently
inflate every metric in the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cardiosense.common.config import load_config  # noqa: E402
from cardiosense.clinical.calibrate import (  # noqa: E402
    PROBABILITY_CLIP,
    _fit_1d_mapping,
    calibrated_confidence,
    clip_probabilities,
    fit_calibrator,
    select_calibration_method,
)
from cardiosense.clinical.data import prepare_dataset  # noqa: E402
from cardiosense.clinical.evaluate import (  # noqa: E402
    evaluate_at_threshold,
    export_errors,
    select_model,
    tune_threshold,
)
from cardiosense.clinical.models import build_logistic_regression, build_xgboost  # noqa: E402
from cardiosense.clinical.predict import ClinicalPredictor  # noqa: E402
from cardiosense.clinical.preprocessing import (  # noqa: E402
    build_preprocessor,
    fit_preprocessor,
    split_data,
    transform_splits,
)

SEED = 42


@pytest.fixture(scope="module")
def cfg():
    return load_config("clinical")


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    """A schema-matched frame: same columns, dtypes and missingness as Cleveland."""
    rng = np.random.default_rng(SEED)
    n = 240
    frame = pd.DataFrame({
        "age": rng.normal(54, 9, n).clip(29, 77).round(),
        "sex": rng.binomial(1, 0.68, n),
        "cp": rng.choice([1, 2, 3, 4], n),
        "trestbps": rng.normal(131, 17, n).clip(94, 200).round(),
        "chol": rng.normal(246, 51, n).clip(126, 564).round(),
        "fbs": rng.binomial(1, 0.15, n),
        "restecg": rng.choice([0, 1, 2], n),
        "thalach": rng.normal(150, 23, n).clip(71, 202).round(),
        "exang": rng.binomial(1, 0.33, n),
        "oldpeak": np.abs(rng.gamma(1.4, 0.75, n)).clip(0, 6.2).round(1),
        "slope": rng.choice([1, 2, 3], n),
        "ca": rng.choice([0.0, 1.0, 2.0, 3.0], n),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    logit = (-1.0 + 0.04 * (frame.age - 54) + 0.9 * frame.sex + 0.8 * (frame.cp == 4)
             + 0.9 * frame.exang + 0.7 * frame.ca + 0.5 * frame.oldpeak
             - 0.02 * (frame.thalach - 150))
    disease = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame["num"] = np.where(disease == 0, 0, rng.choice([1, 2, 3, 4], n))
    frame.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    frame.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return frame


@pytest.fixture(scope="module")
def prepared(raw_frame, cfg):
    return prepare_dataset(raw_frame, cfg)


@pytest.fixture(scope="module")
def splits(prepared, cfg):
    X, y, _ = prepared
    return split_data(X, y, cfg)


@pytest.fixture(scope="module")
def transformed(splits, cfg):
    preprocessor = fit_preprocessor(build_preprocessor(cfg), splits.X_train, splits.y_train)
    X_train, X_val, X_test, names = transform_splits(preprocessor, splits)
    return preprocessor, X_train, X_val, X_test, names


# --------------------------------------------------------------------- data
def test_target_is_binarised_from_severity(prepared):
    _X, y, report = prepared
    assert set(np.unique(y)) <= {0, 1}
    assert "num > 0" in report["target_rule"]


def test_prepare_drops_duplicates(raw_frame, cfg):
    doubled = pd.concat([raw_frame, raw_frame.iloc[:5]], ignore_index=True)
    _X, _y, report = prepare_dataset(doubled, cfg)
    assert report["duplicates_dropped"] >= 5


def test_zero_cholesterol_becomes_missing(raw_frame, cfg):
    """A recorded cholesterol of 0 is a missing-data sentinel, not a measurement."""
    frame = raw_frame.copy()
    frame.loc[frame.index[:7], "chol"] = 0
    X, _y, report = prepare_dataset(frame, cfg)
    assert report["sentinel_zeros_to_nan"]["chol"] == 7
    assert X["chol"].isna().sum() >= 7
    assert (X["chol"] == 0).sum() == 0


def test_missing_column_raises_clear_error(raw_frame, cfg):
    with pytest.raises(KeyError, match="thal"):
        prepare_dataset(raw_frame.drop(columns=["thal"]), cfg)


# -------------------------------------------------------------------- split
def test_split_has_no_overlap(splits):
    """Any overlap between splits is leakage and invalidates every metric."""
    assert sum(splits.summary["index_overlap"].values()) == 0
    train, val, test = (set(splits.X_train.index), set(splits.X_val.index),
                        set(splits.X_test.index))
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


def test_split_sizes_are_fractions_of_the_full_dataset(splits, cfg):
    total = splits.summary["sizes"]["total"]
    assert abs(splits.summary["sizes"]["test"] / total - cfg.split.test_size) < 0.03
    assert abs(splits.summary["sizes"]["val"] / total - cfg.split.val_size) < 0.03


def test_split_preserves_class_balance(splits):
    rates = splits.summary["positive_rate"]
    for name in ("train", "val", "test"):
        assert abs(rates[name] - rates["overall"]) < 0.12, f"{name} split is unbalanced"


def test_split_is_reproducible(prepared, cfg):
    X, y, _ = prepared
    first = split_data(X, y, cfg)
    second = split_data(X, y, cfg)
    assert list(first.X_test.index) == list(second.X_test.index)


# ------------------------------------------------------------ preprocessing
def test_preprocessor_is_fit_on_train_only(splits, cfg):
    """The scaler must learn the TRAIN mean, not the full-dataset mean.

    This is the canonical tabular leak: fitting the transformer on everything
    before splitting lets test statistics shape the training representation.
    """
    train_fitted = fit_preprocessor(build_preprocessor(cfg), splits.X_train)
    all_data = pd.concat([splits.X_train, splits.X_val, splits.X_test])
    all_fitted = build_preprocessor(cfg).fit(all_data)

    train_mean = train_fitted.named_transformers_["numeric"].named_steps["scale"].mean_
    all_mean = all_fitted.named_transformers_["numeric"].named_steps["scale"].mean_

    assert not np.allclose(train_mean, all_mean), (
        "Preprocessor statistics are identical whether fit on train or on everything — "
        "the leakage test cannot detect a leak, so it is not testing anything."
    )

    # The fitted mean must equal the mean of the median-imputed TRAINING columns,
    # and nothing else.
    numeric = list(cfg.dataset.numeric_features)
    train_numeric = splits.X_train[numeric]
    expected = train_numeric.fillna(train_numeric.median()).mean().to_numpy()
    assert np.allclose(train_mean, expected, rtol=1e-6)


def test_transform_produces_named_columns(transformed):
    _pre, X_train, X_val, X_test, names = transformed
    assert X_train.shape[1] == len(names)
    assert X_val.shape[1] == len(names)
    assert X_test.shape[1] == len(names)
    assert not np.isnan(X_train).any(), "Imputation left NaNs in the training matrix"


def test_unseen_category_does_not_crash_inference(transformed, splits, cfg):
    """A new patient with an unseen category code must not blow up at inference."""
    preprocessor = transformed[0]
    row = splits.X_test.iloc[[0]].copy()
    row["thal"] = 99.0  # a code never present in training
    out = preprocessor.transform(row)
    assert out.shape[1] == transformed[1].shape[1]


# ------------------------------------------------------------------- models
def test_logistic_regression_builds_without_deprecation_warnings(cfg):
    import warnings

    X = np.random.default_rng(0).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        build_logistic_regression(cfg).fit(X, y)


def test_xgboost_builds(cfg):
    model = build_xgboost(cfg)
    assert model.get_params()["random_state"] == cfg.seed


# --------------------------------------------------------------- evaluation
def test_select_model_prefers_simpler_on_a_near_tie(cfg):
    results = {
        "logistic_regression": {"val": {"roc_auc": 0.860}},
        "xgboost": {"val": {"roc_auc": 0.865}},   # ahead, but inside tie_tolerance
    }
    selected, decision = select_model(results, cfg)
    assert selected == "logistic_regression"
    assert decision["tie_broken"] is True


def test_select_model_picks_the_clear_winner(cfg):
    results = {
        "logistic_regression": {"val": {"roc_auc": 0.70}},
        "xgboost": {"val": {"roc_auc": 0.88}},
    }
    selected, decision = select_model(results, cfg)
    assert selected == "xgboost"
    assert decision["tie_broken"] is False


def test_threshold_pooled_tuning_uses_more_samples_than_validation_alone(cfg):
    rng = np.random.default_rng(0)
    y_val = rng.integers(0, 2, 45)
    p_val = np.clip(y_val * 0.4 + rng.normal(0.3, 0.2, 45), 0.01, 0.99)
    y_oof = rng.integers(0, 2, 210)
    p_oof = np.clip(y_oof * 0.4 + rng.normal(0.3, 0.2, 210), 0.01, 0.99)

    _threshold, info = tune_threshold(y_val, p_val, cfg, oof_y=y_oof, oof_prob=p_oof)
    assert info["n_tuning_samples"] == 255
    assert "validation_only_alternative" in info


def test_threshold_falls_back_to_validation_without_oof(cfg):
    rng = np.random.default_rng(1)
    y_val = rng.integers(0, 2, 50)
    p_val = rng.uniform(0, 1, 50)
    _threshold, info = tune_threshold(y_val, p_val, cfg)
    assert info["tuned_on"] == "validation"


def test_evaluate_at_threshold_records_the_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = evaluate_at_threshold(y, p, 0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["recall"] == 1.0


def test_export_errors_writes_files_and_counts(tmp_path, splits):
    rng = np.random.default_rng(3)
    y = splits.y_test.to_numpy()
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, len(y)), 0.01, 0.99)
    summary = export_errors(splits.X_test, y, p, 0.5, tmp_path, prefix="test")
    assert (tmp_path / "test_errors.csv").exists()
    assert (tmp_path / "test_predictions.csv").exists()
    assert summary["n_errors"] == sum(summary["counts"].get(k, 0) for k in ("FP", "FN"))


# -------------------------------------------------------------- calibration
def test_sigmoid_mapping_is_monotonic():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.01, 0.99, 400)
    y = rng.binomial(1, p)
    mapping = _fit_1d_mapping("sigmoid", p, y)
    grid = np.linspace(0.01, 0.99, 50)
    out = mapping(grid)
    assert np.all(np.diff(out) >= -1e-9), "Calibration must preserve the ranking"


def test_calibration_selection_returns_a_valid_method(transformed, splits, cfg):
    _pre, X_train, _X_val, _X_test, _names = transformed
    model = build_logistic_regression(cfg).fit(X_train, splits.y_train.to_numpy())
    method, report = select_calibration_method(model, X_train, splits.y_train.to_numpy(), cfg)
    assert method in {"sigmoid", "isotonic"}
    assert report["n_oof_pairs"] == len(splits.y_train)
    # Both candidates must have been scored on held-out pairs, not their own fit data.
    assert set(report["candidates"]) == {"sigmoid", "isotonic"}


def test_calibrator_does_not_refit_the_base_model(transformed, splits, cfg):
    _pre, X_train, X_val, _X_test, _names = transformed
    model = build_logistic_regression(cfg).fit(X_train, splits.y_train.to_numpy())
    coef_before = model.coef_.copy()
    fit_calibrator(model, X_val, splits.y_val.to_numpy(), method="sigmoid")
    assert np.allclose(coef_before, model.coef_), "Base model was refit on calibration data"


def test_probabilities_are_clipped_away_from_certainty():
    assert clip_probabilities(1.0) == PROBABILITY_CLIP[1]
    assert clip_probabilities(0.0) == PROBABILITY_CLIP[0]
    assert np.allclose(clip_probabilities(np.array([0.0, 0.5, 1.0])),
                       [PROBABILITY_CLIP[0], 0.5, PROBABILITY_CLIP[1]])


def test_calibrated_confidence_is_distance_from_the_coin_flip():
    assert calibrated_confidence(0.9) == pytest.approx(0.9)
    assert calibrated_confidence(0.1) == pytest.approx(0.9)   # confident "no"
    assert calibrated_confidence(0.5) == pytest.approx(0.5)   # maximally unsure


# ---------------------------------------------------------------- inference
def test_predictor_round_trip(tmp_path, transformed, splits, cfg):
    """Save artifacts the way train.py does, reload them, and predict."""
    from cardiosense.common.io_utils import save_json, save_pickle

    preprocessor, X_train, X_val, _X_test, names = transformed
    model = build_logistic_regression(cfg).fit(X_train, splits.y_train.to_numpy())
    calibrator = fit_calibrator(model, X_val, splits.y_val.to_numpy(), method="sigmoid")

    save_pickle(model, tmp_path / cfg.output.model_file)
    save_pickle(preprocessor, tmp_path / cfg.output.preprocessor_file)
    save_pickle(calibrator, tmp_path / cfg.output.calibrator_file)
    save_json({
        "model_version": cfg.output.model_version,
        "selected_model": "logistic_regression",
        "threshold": 0.45,
        "label_mapping": {"0": "no disease", "1": "disease present"},
        "raw_features": {"order": list(splits.X_train.columns)},
        "encoded_feature_names": names,
        "calibration": {"method": "sigmoid"},
        "dataset": {"target_rule": "num > 0 -> 1"},
    }, tmp_path / cfg.output.metadata_file)

    predictor = ClinicalPredictor.load(models_dir=tmp_path, cfg=cfg)
    result = predictor.predict_one(splits.X_test.iloc[0].to_dict())

    for key in ("prediction", "probability", "calibrated_probability",
                "calibrated_confidence", "threshold", "model_version"):
        assert key in result
    assert result["prediction"] in {0, 1}
    assert PROBABILITY_CLIP[0] <= result["calibrated_probability"] <= PROBABILITY_CLIP[1]
    assert 0.5 <= result["calibrated_confidence"] <= 1.0
    assert result["model_version"] == cfg.output.model_version


def test_predictor_handles_missing_features(tmp_path, transformed, splits, cfg):
    from cardiosense.common.io_utils import save_json, save_pickle

    preprocessor, X_train, _X_val, _X_test, names = transformed
    model = build_logistic_regression(cfg).fit(X_train, splits.y_train.to_numpy())
    save_pickle(model, tmp_path / cfg.output.model_file)
    save_pickle(preprocessor, tmp_path / cfg.output.preprocessor_file)
    save_json({"model_version": "t", "threshold": 0.5,
               "raw_features": {"order": list(splits.X_train.columns)}},
              tmp_path / cfg.output.metadata_file)

    predictor = ClinicalPredictor.load(models_dir=tmp_path, cfg=cfg)
    partial = {"age": 60, "sex": 1, "cp": 4, "ca": 2, "thal": 7}
    result = predictor.predict_one(partial)
    assert result["prediction"] in {0, 1}


def test_predictor_reports_missing_artifacts_clearly(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="clinical.train"):
        ClinicalPredictor.load(models_dir=tmp_path, cfg=cfg)
