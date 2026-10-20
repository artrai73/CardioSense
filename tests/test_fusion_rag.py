"""Regression tests for the fusion and RAG layers.

These pin *behaviours*, not simulation outputs. A test that asserts an exact
fused score would break on any seed change without indicating anything is
actually wrong; a test that asserts a missing modality carries no weight catches
a real regression.

Run:  python -m pytest tests/test_fusion_rag.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cardiosense.fusion import (  # noqa: E402
    FusionEngine,
    ModalityPrediction,
    build_strategy,
    compute_reliability_indicator,
    detect_disagreement,
    prior_correct,
    reliability_from_ece,
)
from cardiosense.rag import (  # noqa: E402
    HashingBackend,
    build_index,
    build_queries,
    chunk_document,
    ExtractiveProvider,
    screen,
)


def mp(name, risk, reliability=0.9, calibrated=True):
    return ModalityPrediction(modality=name, risk_score=risk,
                              reliability=reliability, is_calibrated=calibrated)


@pytest.fixture
def engine():
    return FusionEngine(adapters={})


# ----------------------------------------------------------- fusion invariants
def test_missing_modality_is_never_imputed(engine):
    """The property the whole design rests on."""
    r = engine.fuse_predictions([
        mp("clinical", 0.86),
        ModalityPrediction.unavailable("ecg"),
        ModalityPrediction.unavailable("xray"),
    ])
    assert "ecg" not in r.fused["weights"]
    assert "xray" not in r.fused["weights"]
    assert r.fused["risk_score"] == pytest.approx(0.86)
    assert set(r.availability["missing"]) == {"ecg", "xray"}


def test_no_modality_available_yields_unknown(engine):
    r = engine.fuse_predictions(
        [ModalityPrediction.unavailable(m) for m in ("clinical", "ecg", "xray")])
    assert r.fused["risk_score"] is None
    assert r.fused["risk_level"] == "UNKNOWN"


def test_failing_modality_does_not_crash_the_assessment(engine):
    r = engine.fuse_predictions([
        mp("clinical", 0.86),
        ModalityPrediction.failed_inference("ecg", "corrupt file"),
        mp("xray", 0.77),
    ])
    assert r.fused["risk_score"] is not None
    assert "ecg" in r.availability["failed"]


def test_all_eight_availability_patterns(engine):
    from itertools import combinations
    names = ("clinical", "ecg", "xray")
    for k in range(4):
        for present in combinations(names, k):
            preds = [mp(n, 0.7) if n in present
                     else ModalityPrediction.unavailable(n) for n in names]
            r = engine.fuse_predictions(preds)
            assert set(r.availability["available"]) == set(present)
            assert set(r.fused["weights"]) <= set(present)
            if not present:
                assert r.fused["risk_score"] is None


# ------------------------------------------------------------- confidence maths
def test_confidence_is_reliability_times_decisiveness():
    p = mp("ecg", 0.9, reliability=0.8)
    assert p.decisiveness == pytest.approx(0.8)
    assert p.confidence == pytest.approx(0.64)


def test_uncalibrated_modality_is_flagged_and_capped(engine):
    r = engine.fuse_predictions([
        mp("clinical", 0.8, calibrated=True),
        mp("ecg", 0.8, reliability=0.5, calibrated=False),
        ModalityPrediction.unavailable("xray"),
    ])
    joined = " ".join(r.warnings).lower()
    assert "uncalibrated" in joined and "ranking scores" in joined


def test_reliability_from_ece_is_monotone():
    assert reliability_from_ece(0.0) == pytest.approx(1.0)
    assert reliability_from_ece(0.125) == pytest.approx(0.5)
    assert reliability_from_ece(0.25) == pytest.approx(0.0)
    assert reliability_from_ece(None) == pytest.approx(0.5)   # unknown != perfect


def test_equal_weighting_is_recovered_at_gamma_zero():
    """A control: gamma=0 must reproduce an unweighted mean exactly."""
    preds = [mp("clinical", 0.9, 0.95), mp("ecg", 0.2, 0.4), mp("xray", 0.6, 0.7)]
    flat = build_strategy("confidence_weighted", gamma=0.0).fuse(preds)
    for weight in flat.weights.values():
        assert weight == pytest.approx(1 / 3, abs=1e-9)


# ------------------------------------------------------------- disagreement
def test_unsure_modalities_are_not_a_conflict():
    """Two modalities saying 'I don't know' must not be flagged."""
    report = detect_disagreement([mp("clinical", 0.4, 0.3), mp("ecg", 0.6, 0.3)])
    assert report.severity == "NONE"


def test_confident_contradiction_is_severe():
    report = detect_disagreement([mp("clinical", 0.95), mp("ecg", 0.05)])
    assert report.severity == "SEVERE"


def test_system_declines_to_adjudicate():
    report = detect_disagreement([mp("clinical", 0.92), mp("ecg", 0.04)])
    assert "does not determine which" in report.interpretation
    assert not any(k in report.to_dict()
                   for k in ("winner", "correct_modality", "wrong_modality"))


def test_single_modality_is_not_assessable_for_agreement():
    """'No comparison was possible' must not be reported as 'they agreed'."""
    report = detect_disagreement([mp("clinical", 0.9)])
    assert report.severity == "NOT ASSESSABLE"
    assert not report.detected


def test_confident_outlier_gains_weight():
    """A documented limitation, pinned so it cannot regress silently.

    Decisiveness is high when a model commits, so a confident dissenter is
    weighted MORE. The safeguard is the disagreement flag and the reduced
    reliability, not the weighting itself.
    """
    strategy = build_strategy("confidence_logodds")
    near = strategy.fuse([mp("clinical", 0.9), mp("ecg", 0.45), mp("xray", 0.85)])
    far = strategy.fuse([mp("clinical", 0.9), mp("ecg", 0.02), mp("xray", 0.85)])
    assert far.weights["ecg"] > near.weights["ecg"]


def test_reliability_falls_with_disagreement():
    preds = [mp("clinical", 0.92), mp("ecg", 0.04), mp("xray", 0.88)]
    agree = 1 - np.mean([abs(a.risk_score - b.risk_score)
                         for a in preds for b in preds if a is not b]) / 2
    high, _ = compute_reliability_indicator(preds, agree, "NONE")
    low, _ = compute_reliability_indicator(preds, agree, "SEVERE")
    assert low < high


# ------------------------------------------------------------- prior correction
def test_prior_correction_preserves_ranking():
    """Strictly monotone, so no ranking and no AUC can change."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    p = np.clip(rng.random(2000) * 0.4 + 0.02, 1e-6, 1 - 1e-6)
    corrected = prior_correct(p, 0.11, 0.025)
    assert spearmanr(p, corrected).statistic > 0.9999
    assert corrected.mean() < p.mean()


def test_prior_correction_clip_does_not_collapse_small_values():
    """The old (0.001, 0.999) clip destroyed ranking among confident negatives."""
    rng = np.random.default_rng(1)
    p = np.clip(rng.random(2000) * 0.3 + 0.01, 1e-6, 1 - 1e-6)
    corrected = prior_correct(p, 0.11, 0.025)
    assert (corrected <= 1e-6 + 1e-12).sum() == 0


# ------------------------------------------------------------------- RAG
@pytest.fixture
def demo_chunks():
    pages = [
        "Header 2021\n\nBlood pressure should be lowered to below 140/90 mmHg in "
        "treated patients. " + "Risk control is central. " * 12,
        "Header 2021\n\nST segment and T wave changes warrant further evaluation. "
        + "Exercise testing may be considered. " * 12,
        "Header 2021\n\nAn enlarged cardiac silhouette may indicate cardiomegaly. "
        + "Echocardiography is the reference investigation. " * 12,
    ]
    return chunk_document(pages, "demo", organization="ESC", year=2021)


def test_chunking_strips_running_headers(demo_chunks):
    assert not any("Header 2021" in c.text for c in demo_chunks)


def test_header_stripping_needs_at_least_three_pages():
    """Documented behaviour: boilerplate cannot be inferred from two samples.

    With fewer than three pages a repeated line is as likely to be content as a
    header, so nothing is removed. Guessing would silently delete real text.
    """
    two = chunk_document(["Hdr\n\n" + "body one. " * 40,
                          "Hdr\n\n" + "body two. " * 40], "d")
    assert any("Hdr" in c.text for c in two)


def test_every_chunk_has_one_page_and_a_citation(demo_chunks):
    for c in demo_chunks:
        assert c.page_number >= 1
        assert "p." in c.citation and "ESC" in c.citation


def test_chunks_respect_the_size_limit(demo_chunks):
    assert max(len(c.text) for c in demo_chunks) <= 1000


def test_non_semantic_backend_is_refused(demo_chunks):
    """It would look like it works and retrieve badly -- a silent failure."""
    with pytest.raises(ValueError, match="not a semantic"):
        build_index(demo_chunks, HashingBackend())


def test_index_search_returns_citations(demo_chunks):
    index = build_index(demo_chunks, HashingBackend(), allow_non_semantic=True)
    hits = index.search("blood pressure target", k=2, min_score=0.0)
    assert hits and all(h.chunk.citation for h in hits)


def test_queries_are_derived_only_from_findings(engine):
    assessment = engine.fuse_predictions([
        mp("clinical", 0.86),
        ModalityPrediction(modality="ecg", risk_score=0.74, reliability=0.5,
                           native_labels={"NORM": 0, "STTC": 1}),
        ModalityPrediction.unavailable("xray"),
    ]).to_dict()
    queries = build_queries(assessment)
    assert queries
    assert all(q.source_finding for q in queries)
    # Nothing about an unavailable modality may appear.
    assert not any("xray" in q.modality for q in queries)


def test_report_keeps_three_content_types_separate(engine):
    assessment = engine.fuse_predictions([mp("clinical", 0.86)]).to_dict()
    report = ExtractiveProvider().generate(assessment, [])
    assert len({s.content_type for s in report.sections}) == 3


def test_extractive_provider_cannot_hallucinate():
    assert ExtractiveProvider().can_hallucinate is False


def test_report_states_when_nothing_is_guideline_derived(engine):
    assessment = engine.fuse_predictions([mp("clinical", 0.86)]).to_dict()
    text = ExtractiveProvider().generate(assessment, []).to_text()
    assert "No guideline evidence was retrieved" in text


# ------------------------------------------------------------- safety screen
@pytest.mark.parametrize("sentence,should_fire", [
    ("This report must not be used for prescribing decisions.", False),
    ("CardioSense does not prescribe treatment.", False),
    ("Fasting blood sugar >120 mg/dl indicates diabetes.", False),
    ("Titrate to 10 mg daily.", True),
    ("The patient has coronary artery disease.", True),
    ("This definitely rules out ischaemia.", True),
    ("No further testing is needed.", True),
])
def test_safety_screen_exemptions_did_not_disarm_it(sentence, should_fire):
    assert bool(screen(sentence)) is should_fire


def test_generated_report_passes_its_own_screen(engine):
    assessment = engine.fuse_predictions([
        mp("clinical", 0.86), mp("ecg", 0.74, 0.5, False), mp("xray", 0.79),
    ]).to_dict()
    text = ExtractiveProvider().generate(assessment, []).to_text()
    assert screen(text) == [], "the system's own report tripped the screen"


# --------------------------------------------------- ECG feature preprocessing
class _ScaleSensitiveBooster:
    """Stands in for one XGBClassifier: NaN-tolerant and scale-sensitive."""

    def __init__(self, weights, bias=-0.5):
        self.w, self.b = np.asarray(weights, dtype=float), float(bias)

    def predict_proba(self, X):
        z = np.nansum(np.asarray(X, dtype=float) * self.w, axis=1) + self.b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - p, p])


@pytest.fixture
def ecg_bundle_dir(tmp_path):
    """Write an ``ecg_feature_model.pkl`` factory parameterised by mode."""
    import pickle

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    cols = [f"f{i}" for i in range(12)]
    rng = np.random.default_rng(3)

    # Fit on a DataFrame, as cell 22 of the training notebook does. sklearn then
    # remembers the column names and complains if a differently ordered frame is
    # passed later -- a safety net the adapter should be exercising, so the
    # fixture must not quietly opt out of it by fitting on a bare array.
    import pandas as pd

    train = pd.DataFrame(rng.normal(50, 20, size=(200, len(cols))), columns=cols)
    train.iloc[:20, :3] = np.nan
    imputer = SimpleImputer(strategy="median").fit(train)
    scaler = StandardScaler().fit(imputer.transform(train))
    boosters = [_ScaleSensitiveBooster(rng.normal(0, 0.05, len(cols)))
                for _ in classes]

    def write(**overrides):
        bundle = {"model": boosters, "feature_cols": cols, "classes": classes,
                  "thresholds": np.full(len(classes), 0.3), "clip_sd": 5.0,
                  "imputer": imputer, "scaler": scaler,
                  "preprocessing": "raw",
                  "metadata": {"model_version": "test-v1", "is_calibrated": False}}
        bundle.update(overrides)
        with open(tmp_path / "ecg_feature_model.pkl", "wb") as handle:
            pickle.dump(bundle, handle)
        return tmp_path

    row = {c: float(v) for c, v in zip(cols, rng.normal(50, 20, len(cols)))}
    row[cols[0]] = np.nan                    # a genuine extractor miss
    write.row = row                          # type: ignore[attr-defined]
    return write


def test_raw_and_scaled_preprocessing_disagree(ecg_bundle_dir):
    """Why the ``preprocessing`` field exists at all.

    ``features_xgb`` is fitted on the raw frame with NaNs intact, because the
    booster learns a default direction for each missing value. Imputing and
    scaling first raises no error -- the medians land inside the range the trees
    already split on -- so the predictions are plausible and wrong. Only the
    recorded mode prevents it, which is worth a test that fails if the two modes
    ever become interchangeable.
    """
    from cardiosense.fusion import ECGAdapter

    row = ecg_bundle_dir.row
    raw = ECGAdapter(ecg_bundle_dir(preprocessing="raw")).predict(row)
    scaled = ECGAdapter(ecg_bundle_dir(preprocessing="impute_scale_clip")).predict(row)

    assert raw.available and scaled.available
    assert abs(raw.risk_score - scaled.risk_score) > 1e-3


def test_bundle_without_preprocessing_field_is_refused(ecg_bundle_dir):
    """An unmarked bundle is ambiguous, and guessing is the silent-failure path."""
    from cardiosense.fusion import ECGAdapter

    adapter = ECGAdapter(ecg_bundle_dir(preprocessing=None))
    assert not adapter.available
    assert "preprocessing" in adapter.load_error


def test_raw_mode_preserves_nan(ecg_bundle_dir):
    """NaNs must reach the booster, not be filled behind its back."""
    from cardiosense.fusion import ECGAdapter

    import pandas as pd

    adapter = ECGAdapter(ecg_bundle_dir(preprocessing="raw"))
    frame = pd.DataFrame([ecg_bundle_dir.row]).reindex(columns=adapter.feature_cols)
    assert np.isnan(adapter._prepare(frame)).any()


def test_one_vs_rest_estimator_count_is_checked(ecg_bundle_dir):
    """Five superclasses need five boosters; three is a truncated save."""
    from cardiosense.fusion import ECGAdapter
    import pickle

    directory = ecg_bundle_dir(preprocessing="raw")
    with open(directory / "ecg_feature_model.pkl", "rb") as handle:
        bundle = pickle.load(handle)
    bundle["model"] = bundle["model"][:3]
    with open(directory / "ecg_feature_model.pkl", "wb") as handle:
        pickle.dump(bundle, handle)

    result = ECGAdapter(directory).predict(ecg_bundle_dir.row)
    assert result.failed and "3 estimators" in result.reason


def test_ecg_adapter_resolution_order_and_reasons(tmp_path):
    """With no artifacts at all, every attempted path must be reported."""
    from cardiosense.fusion import ECGSignalAdapter, resolve_ecg_adapter

    adapter = resolve_ecg_adapter(tmp_path, search_roots=[tmp_path])
    assert isinstance(adapter, ECGSignalAdapter)
    assert not adapter.available
    for name in ("ECGAdapter", "ECGJointAdapter", "ECGSignalAdapter"):
        assert name in adapter.load_error


# ------------------------------------------------ ECG checkpoint (.pt) loading
torch = pytest.importorskip("torch", reason="checkpoint loading needs torch")


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Write checkpoints in the exact layout ``train_joint()`` saves."""
    from cardiosense.fusion.ecg_joint import build_joint_net

    n_feat = 24
    names = [f"f{i}" for i in range(n_feat)]
    directory = tmp_path / "models" / "ecg_fusion"
    directory.mkdir(parents=True)

    torch.manual_seed(0)
    last = build_joint_net(n_feat, 5)
    torch.manual_seed(1)
    best = build_joint_net(n_feat, 5)

    for seed in (42, 43):
        torch.save({
            "model": last.state_dict(),
            "opt": {}, "sched": {}, "scaler": {},
            "state": {"epoch": 17, "best_auc": 0.9287, "bad": 3, "done": True,
                      "n_feat": n_feat, "feature_names": names,
                      "best_state": {k: v.detach().clone()
                                     for k, v in best.state_dict().items()}},
        }, directory / f"joint_seed{seed}.pt")

    return tmp_path, n_feat, best


def test_checkpoint_loader_prefers_best_state_over_last_epoch(checkpoint_dir):
    """``ckpt["model"]`` is the last epoch, including ones early stopping rejected.

    Loading it instead of ``state["best_state"]`` costs accuracy and raises
    nothing, anywhere.
    """
    from cardiosense.fusion.ecg_joint import find_joint_checkpoints, load_joint_checkpoint

    root, _, best = checkpoint_dir
    path = find_joint_checkpoints([root])[0]

    loaded = load_joint_checkpoint(path)
    assert loaded.weights_key == "state.best_state"
    reference = best.state_dict()
    for key, value in loaded.model.state_dict().items():
        assert torch.allclose(value.float(), reference[key].float())

    other = load_joint_checkpoint(path, prefer_best=False)
    assert other.weights_key == "model"
    assert not torch.allclose(loaded.model.state_dict()["head.1.weight"],
                              other.model.state_dict()["head.1.weight"]), \
        "both keys hold the same weights, so this test proves nothing"


def test_foreign_checkpoint_fails_loudly(checkpoint_dir, tmp_path):
    from cardiosense.fusion.ecg_joint import build_joint_net, load_joint_checkpoint

    path = tmp_path / "foreign.pt"
    torch.save({"state_dict": build_joint_net(24, 5).state_dict()}, path)
    with pytest.raises(KeyError, match="expected the"):
        load_joint_checkpoint(path)


def test_mismatched_seed_is_skipped_not_fatal(checkpoint_dir):
    """A stale checkpoint from another run must not take the ensemble down.

    The feature count is read from ``state`` before the model is built. Checking
    it afterwards never ran -- ``load_state_dict`` had already raised.
    """
    from cardiosense.fusion.ecg_joint import (
        build_joint_net, find_joint_checkpoints, load_joint_ensemble,
    )

    root, n_feat, _ = checkpoint_dir
    stray = build_joint_net(n_feat + 7, 5)
    (root / "models" / "ecg_fusion" / "joint_seed99.pt").write_bytes(b"")
    torch.save({"model": stray.state_dict(), "opt": {}, "sched": {}, "scaler": {},
                "state": {"epoch": 2, "best_auc": 0.5, "done": False, "bad": 0,
                          "n_feat": n_feat + 7, "feature_names": [],
                          "best_state": stray.state_dict()}},
               root / "models" / "ecg_fusion" / "joint_seed99.pt")

    paths = find_joint_checkpoints([root])
    assert len(paths) == 3
    with pytest.warns(RuntimeWarning, match="skipping"):
        loaded = load_joint_ensemble(paths)
    assert len(loaded) == 2
    assert {c.n_feat for c in loaded} == {n_feat}


def test_signal_only_needs_no_preprocessor(checkpoint_dir):
    """The property that makes this the always-available ECG path."""
    from cardiosense.fusion import ECGSignalAdapter

    root, _, _ = checkpoint_dir
    adapter = ECGSignalAdapter(root / "models" / "ecg", search_roots=[root])
    assert adapter.available, adapter.load_error

    rng = np.random.default_rng(0)
    prediction = adapter.predict(rng.normal(0, 0.1, size=(12, 1000)))
    assert prediction.available
    assert 0.0 <= prediction.risk_score <= 1.0
    assert prediction.is_calibrated is False
    assert prediction.reliability == pytest.approx(0.50)
    assert adapter.status()["model_version"], "status must name the model that ran"


def test_untuned_thresholds_are_declared_not_hidden(checkpoint_dir):
    """0.5 is wrong for every PTB-XL superclass, so the fallback must be visible."""
    from cardiosense.fusion import ECGSignalAdapter

    root, _, _ = checkpoint_dir
    adapter = ECGSignalAdapter(root / "models" / "ecg", search_roots=[root])
    assert np.allclose(adapter.thresholds, 0.5)
    assert "not tuned" in adapter.metadata["threshold_source"]


def test_lead_orientation_is_handled_not_guessed(checkpoint_dir):
    """(1000, 12) must transpose; 7 leads must raise rather than run."""
    from cardiosense.fusion.ecg_joint import (
        find_joint_checkpoints, load_joint_ensemble, predict_signal_only,
    )

    root, _, _ = checkpoint_dir
    ensemble = load_joint_ensemble(find_joint_checkpoints([root]))
    rng = np.random.default_rng(1)
    wave = rng.normal(0, 0.1, size=(12, 1000)).astype(np.float32)

    assert np.allclose(predict_signal_only(ensemble, wave),
                       predict_signal_only(ensemble, wave.T), atol=1e-6)
    with pytest.raises(ValueError, match="12 ECG leads"):
        predict_signal_only(ensemble, rng.normal(size=(7, 1000)))


def test_batch_prediction_matches_single_record(checkpoint_dir):
    from cardiosense.fusion.ecg_joint import (
        find_joint_checkpoints, load_joint_ensemble, predict_signal_batch,
        predict_signal_only,
    )

    root, _, _ = checkpoint_dir
    ensemble = load_joint_ensemble(find_joint_checkpoints([root]))
    rng = np.random.default_rng(2)
    waves = rng.normal(0, 0.1, size=(3, 12, 1000)).astype(np.float32)

    batched = predict_signal_batch(ensemble, waves, batch_size=2)
    assert batched.shape == (3, 5)
    for i in range(3):
        assert np.allclose(batched[i], predict_signal_only(ensemble, waves[i]),
                           atol=1e-5)


def test_joint_refuses_mismatched_waveform_and_feature_counts(checkpoint_dir):
    """Pairing N ECGs with M feature rows would silently mix patients."""
    from cardiosense.fusion.ecg_joint import (
        find_joint_checkpoints, load_joint_ensemble, predict_joint_batch,
    )

    root, n_feat, _ = checkpoint_dir
    ensemble = load_joint_ensemble(find_joint_checkpoints([root]))
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="same records in the same order"):
        predict_joint_batch(ensemble,
                            rng.normal(size=(3, 12, 1000)).astype(np.float32),
                            rng.normal(size=(2, n_feat)).astype(np.float32))


def test_ensemble_averages_in_logit_space(checkpoint_dir):
    """Matching the notebook's ``mean_logit``; averaging probabilities would
    pull the ensemble toward 0.5 and invalidate every tuned threshold."""
    from cardiosense.fusion.ecg_joint import mean_logit

    eps = 1e-6
    rng = np.random.default_rng(4)
    a, b = rng.random((4, 5)), rng.random((4, 5))

    def reference(probs):
        logits = [np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
                  for p in probs]
        return 1 / (1 + np.exp(-np.mean(logits, axis=0)))

    assert np.allclose(mean_logit([a, b]), reference([a, b]))
    assert not np.allclose(mean_logit([a, b]), np.mean([a, b], axis=0))
