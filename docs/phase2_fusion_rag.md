# CardioSense Phase 2 — Fusion and Guideline Retrieval

Drop-in code and notebooks that combine your three trained Phase 1 models and
ground the output in clinical guidelines.

**48 tests pass. All notebooks were executed or syntax-verified before packaging.**

---

## Read this first: what the ECG notebook did and did not save

`04_ecg_feature_fusion.ipynb` saves **more than `summary.json`**. Its
`train_joint()` calls `torch.save` at the end of every epoch, so
`joint_seed{seed}.pt` is on Drive and survives a disconnect.

| artifact | in the `.pt`? | recovery |
|---|---|---|
| `GatedJointNet` weights (best epoch) | **yes** — `state["best_state"]` | load |
| signal-only head (`signal_seresnet`) | **yes** — same weights | load |
| feature column order, `n_feat` | **yes** — `state["feature_names"]` | load |
| best val macro AUC, epoch count | **yes** — `state` | load |
| imputer / scaler | no | refit on train, no training |
| per-class thresholds | no | re-tune on val |
| `features_xgb` boosters (**the selected model**) | no | re-run one cell, 1–3 min |

So nothing is unrecoverable, but two of those rows cost a step, and one of them
is the model notebook 04 actually selected.

### Two subtleties in the checkpoint format

**`ckpt["model"]` is not the best model.** The save runs every epoch, so that key
holds whatever the *last* epoch produced — including the epochs early stopping
had already rejected. The weights to use are `ckpt["state"]["best_state"]`.
Loading the wrong one costs accuracy and raises nothing, anywhere.
`load_joint_checkpoint()` prefers `best_state` and records which key it used.

**The signal branch has no preprocessing dependency.** `signal_logits()` runs the
encoder and its own auxiliary head and never enters the feature branch, so
waveform-only inference works from the `.pt` files and nothing else. That is the
one ECG path that works from a cold runtime with no further steps.

---

## Install

```bash
cp -r src/cardiosense/fusion  <your-repo>/src/cardiosense/
cp -r src/cardiosense/rag     <your-repo>/src/cardiosense/
cp -r notebooks/*.ipynb       <your-repo>/notebooks/
cp tests/test_fusion_rag.py   <your-repo>/tests/

pip install sentence-transformers pypdf        # RAG only
python -m pytest tests/test_fusion_rag.py -q   # expect 48 passed
```

Nothing in Phase 1 is modified. The fusion layer *wraps* your trained models
and never retrains or alters them.

---

## Run order

Pick **one** of 04b or 04c depending on whether notebook 04's kernel is alive.

| # | notebook | needs | time |
|---|---|---|---|
| 1a | `04b_save_ecg_feature_model.ipynb` | notebook 04's kernel still alive | 1 min |
| 1b | `04c_recover_ecg_from_checkpoints.ipynb` | the `.pt` files only (+ 04 §1–§8 for Part B) | 5 min |
| 2 | `05_multimodal_fusion.ipynb` | whatever models are on Drive | 5 min |
| 3 | `06_rag_guidelines.ipynb` | one guideline PDF (optional) | 10 min |

Notebooks 2 and 3 run even when models or PDFs are missing — they report what
they had rather than failing.

`04c` is split deliberately: **Part A is a cold start** (loads and verifies the
checkpoints, needs nothing from notebook 04), and **Part B** needs notebook 04's
feature pipeline in memory — run 04 from the top **through §8 and stop**, which
downloads two CSVs and fits an imputer but trains nothing.

---

## The three ECG paths

`resolve_ecg_adapter()` tries them in this order and `status()["model_version"]`
names whichever loaded, so a report can never imply the selected model produced
a number the fallback produced.

| adapter | needs | payload |
|---|---|---|
| `ECGAdapter` (`features_xgb`) | `ecg_feature_model.pkl` | raw feature row |
| `ECGJointAdapter` (`joint_gated`) | `.pt` + `ecg_feature_preprocessor.pkl` | `{waveform, features}` |
| `ECGSignalAdapter` (`signal_only`) | the `.pt` files alone | `(12, 1000)` array |

### The bug this ordering exposed

`features_xgb` was fitted on **`X_train_raw`** — the raw frame, NaNs intact,
because XGBoost learns a default direction for each missing value. The MLP and
the joint network's feature branch were fitted on the **imputed, scaled, clipped**
matrix. My first adapter imputed and scaled before calling the boosters.

Feeding scaled values to those boosters raises **no error**: the medians land
inside the range the trees already split on, so every prediction is plausible
and wrong. On one validation row with a stand-in booster the two modes gave
**0.0066 and 0.6350** for the same patient.

So the bundle now records `preprocessing` (`"raw"` or `"impute_scale_clip"`),
`ECGAdapter` honours it, and a bundle that does not declare a mode **refuses to
load** rather than guessing. `test_raw_and_scaled_preprocessing_disagree` fails
if the two modes ever become interchangeable, which would make the flag
pointless and the test a lie.

---

## Fusion: how it works

### A common scale, written down

| modality | native output | mapped risk |
|---|---|---|
| clinical | P(stenosis >50%) | used directly, calibrated |
| ECG | 5 superclass probabilities | `1 - P(NORM)` |
| X-ray | P(cardiomegaly) | prior-corrected to population prevalence |

`1 - P(NORM)` rather than a max over abnormal classes, because NORM is an
explicitly trained class — the model's own estimate that the trace is not
normal, not an ad-hoc aggregation.

### Confidence

```
confidence = reliability × decisiveness
```

* **reliability** = `1 − ECE/0.25` on held-out data. Model-level. Capped at
  **0.50** for any uncalibrated modality — which, today, means ECG and X-ray.
  Every ECG path reports `is_calibrated=False`, so that cap applies whichever
  one loads. Running `cardiosense.ecg.calibrate` is what lifts it honestly.
* **decisiveness** = `|2p − 1|`. Case-specific.

Both terms are needed. Without reliability an overconfident model dominates, and
`pos_weight` training pushes the X-ray model toward extremes by construction.
Without decisiveness the fusion cannot adapt per patient.

### Three invariants, asserted in code

1. A missing modality carries **no weight** — excluded, never imputed.
2. A failing modality does not crash the assessment.
3. No modality available yields **UNKNOWN**, not a fabricated number.

### Measured on the synthetic testbed

| strategy | ROC-AUC | Brier | ECE |
|---|---|---|---|
| mean | 0.6569 | 0.2348 | 0.0468 |
| max | 0.6431 | 0.2406 | 0.0705 |
| noisy_or | 0.6158 | 0.3465 | **0.3225** |
| confidence_weighted | 0.6491 | 0.2330 | 0.0080 |
| **confidence_logodds** | 0.6489 | **0.2330** | **0.0077** |

Two things to take from this table:

**Noisy-OR is 0.04 AUC below mean fusion and has a 7× worse ECE.** It
assumes conditionally independent detectors, and three views of one heart are
not independent. A strategy cannot be chosen on discrimination alone.

**Confidence weighting costs ~0.008 AUC and cuts ECE by ~84%.** That is the
honest trade, and it is why the claim is *better-calibrated*, not *more
accurate*.

### The limitation that is pinned, not hidden

A confident outlier gains weight, because decisiveness is high precisely when a
model commits. `test_confident_outlier_gains_weight` locks this so it cannot
regress silently.

A median or trimmed pool would resist outliers — and would discard exactly the
minority report a clinician most needs to see. The safeguard chosen is the
disagreement flag and the reduced reliability indicator beside it.

---

## RAG: how it works

```
assessment → queries (from findings only) → exact search → verbatim excerpts → report
```

**The default generator is extractive.** It copies guideline sentences verbatim
with citations and composes nothing, so it **cannot** hallucinate — not "is
unlikely to", cannot.

**Hallucination would enter at query construction, not generation.** If a query
could invent "chest pain on exertion" because the risk came out high, retrieval
would faithfully fetch evidence for a patient who does not exist. So
`build_queries()` reads model findings and nothing else, and each query records
the finding that produced it.

**Exact search, not approximate.** A few thousand chunks scan in under a
millisecond. An approximate index would mean a passage could be *silently
missed*, in a system whose purpose is grounding claims in retrievable text.

**Three content types, always separate:** `AI MODEL FINDING`,
`GUIDELINE EVIDENCE`, `GENERATED SUMMARY`. Conflating them is the central safety
risk — a clinician who cannot tell a model's guess from a guideline's
recommendation has lost the ability to weigh either.

### The safety screen, and its two awkward exemptions

Every emitted string is screened for diagnostic assertion, prescription, drug
doses and false certainty — with a 200-character prohibition window, because:

* *"must not be used for **prescribing** decisions"* forbids the act it names
* *">120 **mg**/dl"* is a lab unit, not a dose (the slash distinguishes it)

A screen that fires on its own disclaimers is worse than no screen: it trains
everyone to ignore the alerts, which is exactly how a real prescriptive sentence
later slips through. Seven cases pin that the exemptions did not disarm it.

---

## What this does *not* claim

The three models are trained on **disjoint cohorts** with **different targets**.
No patient appears in more than one. There is no dataset on which "fusion
improves diagnostic accuracy" can be measured for this system, and none is
claimed.

Synthetic-testbed numbers validate the **mechanism** against known generative
parameters. They are not clinical performance.

Other stated limitations:

- The fused output is **not itself calibrated**. Averaging calibrated inputs
  degrades calibration; log-odds pooling reduces but does not remove this.
- Reliability weights (0.40 / 0.25 / 0.25 / 0.10), risk bands (0.33 / 0.66) and
  disagreement cut-points are **project conventions, not clinical standards**.
- The safety screen is pattern matching. A floor, not a guarantee.
- Guideline editions must be verified against the issuing body before citing.

---

## For the review

> Three models trained on disjoint cohorts are mapped onto a documented common
> scale and combined by confidence-weighted log-odds pooling, where confidence
> is reliability (from held-out calibration error) times decisiveness (distance
> from a coin flip). Missing modalities are excluded rather than imputed;
> disagreement is detected, flagged, and deliberately not adjudicated. Guideline
> evidence is retrieved by exact semantic search and quoted verbatim with page
> citations, and model findings, guideline text and generated narrative are
> labelled separately throughout. On a controlled testbed, confidence weighting
> reduces fused calibration error by roughly 84% relative to mean fusion at a
> cost of 0.008 AUC. No clinical accuracy claim is made, because no paired
> multimodal cohort exists for these models.

---

## Integration notes (2026-10-20)

Integrated into `CardioSense-main` and verified on a CPU environment
(torch 2.14, scikit-learn 1.9, xgboost 3.4):

* **166 tests pass**: the 111 Phase 1 tests, the 48 shipped with this package,
  and 7 new ones in `tests/test_fusion_phase1_artifacts.py`.
* Notebooks 05 and 06 execute end to end with no models or PDFs present
  (degraded mode); the strategy table reproduces the numbers above exactly.
* `ecg_joint.GatedJointNet` is identical to notebook 04's (183 tensors, same
  shapes); a checkpoint in `train_joint()`'s format reloads with bit-identical
  signal-only and joint predictions.

The shipped tests use stand-in clinical and X-ray models, so four mismatches
with the real Phase 1 artifacts went unseen. The new test file builds artifacts
through the Phase 1 code paths and pins each fix:

| # | adapter | problem with real Phase 1 artifacts | fix |
|---|---|---|---|
| 1 | all | Phase 1 `save_pickle` uses `joblib.dump`; `pickle.load` raised `UnpicklingError`, so the clinical model never loaded | `_read_pickle` uses `joblib.load` (reads plain pickles too) |
| 2 | clinical | the calibrator is a `CalibratedClassifierCV` around the model and needs the feature matrix; it was given the raw probability, failing every prediction (`X has 1 features...`) | call it on the preprocessed matrix, as `clinical/predict.py` does |
| 3 | clinical | held-out ECE lives at `calibration.metrics.calibrated.ece`; it was not found, so a calibrated model was capped at reliability 0.50 | read that key first |
| 4 | X-ray | a failed strict load fell back to `strict=False` with stripped keys, which loads nothing and returns a random network's output | build `XrayDenseNet121` itself and load strictly |
| 5 | X-ray | images converted straight to RGB; Phase 1 reads grey and replicates channels (differs for RGBA files) | same transform as `xray/predict.py` |
| 6 | X-ray | Phase 1 writes no population prevalence, so prior correction never ran, yet the output was labelled population-referenced | the label appears only when the correction runs; `status()["prior_correction"]` says which. Add `population_prevalence_for_prior_correction` to `xray_config.json` to enable it |

Notebook 05 also saved the strategy table under `disagreement_sweep` (a reused
`rows` variable); fixed.

**On Colab:** the notebooks import `cardiosense.fusion` and `cardiosense.rag`
from the cloned repository, so these files must be pushed to the GitHub repo
before 04c, 05 or 06 will run there.

**Run order on Colab (simplest):** run `04_ecg_feature_fusion.ipynb` with its main
settings. Its section 15 (the 04b cells, appended because Colab notebooks do not
share a kernel) saves and verifies the ECG bundles; ablation runs skip it. Then
run 05, then 06. `04c` remains the cold-start path from the `.pt` files alone.
