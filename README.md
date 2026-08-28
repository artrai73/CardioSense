# CardioSense

**A Confidence-Aware Multi-Modal Explainable Clinical Decision Support System for Cardiovascular Disease using Guideline-Grounded AI**

B.Tech final-year project — **Phase 1**

---

## 1. Project overview

CardioSense is a clinical decision support system for cardiovascular disease that
combines three sources of patient evidence — structured clinical variables, the
12-lead ECG, and the chest radiograph — into explainable, confidence-aware risk
assessment.

The full system (Phase 2) fuses the three modalities with uncertainty-weighted
confidence, grounds its recommendations in retrieved clinical guidelines, and
generates a clinician-facing report. **This repository currently implements
Phase 1: the three predictive pipelines themselves**, each trained, evaluated,
calibrated where applicable, and explained independently.

---

## 1a. Phase 1 status: COMPLETE

All three pipelines are implemented, tested and runnable end to end.

| Pipeline | Dataset | Task | Headline metric | Explainability | Notebook |
|---|---|---|---|---|---|
| Clinical | UCI Heart Disease | Binary risk | ROC-AUC + Brier | SHAP | `01_clinical_training.ipynb` |
| ECG | PTB-XL | 5-class, **multi-label** | Macro ROC-AUC | Integrated Gradients | `02_ecg_training.ipynb` |
| X-ray | NIH ChestX-ray14 | Binary cardiomegaly | **PR-AUC** | Grad-CAM | `03_xray_training.ipynb` |

111 tests pass on CPU with no datasets present:

```bash
python -m pytest tests/ -q
```

Each pipeline also runs headless:

```bash
python -m cardiosense.clinical.train
python -m cardiosense.ecg.train
python -m cardiosense.xray.train
```

## 2. Phase 1 scope

> **Multimodal fusion, confidence-aware fusion, RAG, LLM-based recommendations,
> clinical report generation, backend integration and the React dashboard are
> intentionally outside Phase 1.**

Phase 1 delivers three completely independent pipelines. Each runs, trains,
evaluates, explains and saves its own model with no dependency on the others.

**In scope**

- Clinical risk prediction: EDA → leakage-free preprocessing → Logistic
  Regression → XGBoost → probability calibration → SHAP
- ECG interpretation: PTB-XL → signal preprocessing → classical baseline →
  1D CNN → (optional ResNet-1D) → Integrated Gradients
- Chest X-ray analysis: patient-level split → augmentation → baselines →
  DenseNet121 transfer learning → Grad-CAM
- Reproducible experiment tracking, error analysis and inference scripts

**Out of scope (Phase 2)**

- Any fusion of the three outputs
- Retrieval-augmented guideline grounding
- LLM recommendation or report generation
- FastAPI backend, React dashboard, deployment

Deliberately, the three `predict.py` scripts return **structured but unfused**
outputs. Confidence calibration in the clinical pipeline exists precisely so that
Phase 2 fusion has a meaningful confidence to weight by.

---

## 3. Phase 1 architecture

```text
                          CARDIOSENSE — PHASE 1


     ┌─────────────────────────────────────────┐
     │              Clinical Data              │
     │        (UCI Heart Disease, n=303)       │
     └────────────────────┬────────────────────┘
                          ↓
                   Preprocessing
              (ColumnTransformer, fit on
               train split only)
                          ↓
                Logistic Regression                  ← Experiment C-A
                          ↓
                       XGBoost                       ← Experiment C-B
                 (RandomizedSearchCV)
                          ↓
                    Calibration                      ← Experiment C-C
             (Platt vs isotonic, chosen
              empirically on validation)
                          ↓
                         SHAP
                (global + per-patient)
                          ↓
                    Saved Model
        clinical_model.pkl / preprocessor.pkl
              / calibrator.pkl


     ┌─────────────────────────────────────────┐
     │                  ECG                    │
     │      (PTB-XL, 12-lead, 10 s, 100 Hz)    │
     └────────────────────┬────────────────────┘
                          ↓
                  Signal Processing
        (0.5 Hz high-pass, per-lead z-score,
             fixed 1000-sample length)
                          ↓
              Statistical + LogReg baseline         ← Experiment E-A
                          ↓
                       1D CNN                        ← Experiment E-B
              (4 conv stages, BN, dropout)
                          ↓
                     ResNet-1D                       ← Experiment E-C
                    (only if justified)
                          ↓
                 ECG Explainability
                (Integrated Gradients over
                 the raw waveform)
                          ↓
                    Saved Model
              ecg_model.pth / ecg_config.json


     ┌─────────────────────────────────────────┐
     │              Chest X-ray                │
     │   (NIH ChestX-ray14, PA, Cardiomegaly)  │
     └────────────────────┬────────────────────┘
                          ↓
              Patient-level split
         (official NIH patient-disjoint lists)
                          ↓
                  Image Processing
          (224px, ImageNet norm, train-only
           augmentation, no horizontal flip)
                          ↓
              Majority + pixel-feature LogReg        ← Experiment X-A
                          ↓
                    DenseNet121                      ← Experiment X-B
             (ImageNet init, head-then-
              partial-unfreeze fine-tuning)
                          ↓
                     Evaluation
              (PR-AUC headline, not accuracy)
                          ↓
                      Grad-CAM
                          ↓
                    Saved Model
             xray_model.pth / xray_config.json


   ─────────────────────────────────────────────────────────────
        NOT IN PHASE 1:  fusion · RAG · LLM reports · dashboard
   ─────────────────────────────────────────────────────────────
```

---

## 4. Datasets

| Modality | Dataset | Access | Target | Task type |
|---|---|---|---|---|
| Clinical | UCI Heart Disease (Cleveland, n=303) | Open, auto-download via `ucimlrepo` | Presence of coronary disease (`num > 0`) | Binary |
| ECG | PTB-XL v1.0.3 (~21.8k records) | Open, PhysioNet direct download | 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP | **Multi-label** |
| X-ray | NIH ChestX-ray14 (112,120 images) | Open, Kaggle mirror | Cardiomegaly | Binary, imbalanced (~2.5%) |

Full rationale — why these datasets, why these targets, why ChestX-ray14 over
CheXpert, and exact download commands — is in **[`docs/datasets.md`](docs/datasets.md)**.

Three decisions worth surfacing here:

- **PTB-XL is multi-label, not multi-class.** One record can be both `MI` and
  `STTC`. Plain accuracy is therefore never reported for the ECG pipeline;
  exact-match ratio is reported instead, under that name.
- **PTB-XL splits use the official `strat_fold` column** (1–8 / 9 / 10), which is
  patient-disjoint by construction. Records are never randomly re-split.
- **X-ray splits are patient-level.** A patient contributes many films; an
  image-level split leaks the same chest across train and test.

---

## 5. Installation

### Local (development, CPU work, tests)

```bash
git clone https://github.com/<your-username>/CardioSense.git
cd CardioSense

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # makes `import cardiosense` work anywhere

python scripts/verify_setup.py
```

### Requirements

- Python ≥ 3.10
- ~2 GB disk for PTB-XL at 100 Hz; ~45 GB transient for the raw ChestX-ray14 download
- A CUDA GPU for the ECG and X-ray pipelines (the clinical pipeline is CPU-only)

---

## 6. Google Colab setup

Open `notebooks/00_colab_setup.ipynb` and run it top to bottom. It:

1. Verifies the GPU (`Runtime → Change runtime type → GPU` if absent)
2. Mounts Google Drive
3. Clones or pulls this repository
4. Installs `requirements-colab.txt` — **not** `requirements.txt`, because Colab
   already ships a CUDA-matched `torch`; reinstalling it is the most common cause
   of "CUDA suddenly unavailable"
5. Sets `CARDIOSENSE_DATA_ROOT` to your Drive folder
6. Runs `scripts/verify_setup.py --check-data`

The only machine-specific setting is the data root:

```python
import os
os.environ["CARDIOSENSE_DATA_ROOT"] = "/content/drive/MyDrive/CardioSense/data"
```

Every other path in the project is resolved relative to the repository root, so
nothing else needs editing between your laptop and Colab.

**Surviving disconnects.** All GPU training writes `last.pt` every epoch and
`best.pt` on improvement, including optimiser, scheduler, AMP scaler and RNG
state. Re-running the training cell after a dropped runtime resumes from the next
epoch rather than restarting.

---

## 7. Dataset preparation

See [`docs/datasets.md`](docs/datasets.md) for the full commands. In short:

| Dataset | Manual step needed? |
|---|---|
| UCI Heart Disease | No — fetched automatically by `ucimlrepo` |
| PTB-XL | Yes — one `wget` of a 1.7 GB zip from PhysioNet (open access) |
| ChestX-ray14 | Yes — Kaggle API token, then `kaggle datasets download -d nih-chest-xrays/data` |

Expected final layout under `$CARDIOSENSE_DATA_ROOT`:

```text
data/
├── clinical/          (auto-populated)
├── ecg/ptbxl/         ptbxl_database.csv, scp_statements.csv, records100/
└── xray/nih/          Data_Entry_2017_v2020.csv, images/, train_val_list.txt, test_list.txt
```

---

## 8. Training commands

### Clinical (implemented)

```bash
# Full pipeline: EDA -> split -> LogReg -> XGBoost -> select -> calibrate -> SHAP
# CPU only, ~2 minutes.
python -m cardiosense.clinical.train

# Faster iteration while developing
python -m cardiosense.clinical.train --skip-eda --set models.xgboost.n_iter_search=10

# Ablations, without editing any YAML
python -m cardiosense.clinical.train --set seed=7
python -m cardiosense.clinical.train --set selection.threshold_tuning_data=validation
python -m cardiosense.clinical.train --set models.logistic_regression.enabled=false
```

Inference:

```bash
python -m cardiosense.clinical.predict --example
python -m cardiosense.clinical.predict --json '{"age": 63, "sex": 1, "cp": 4, "ca": 2, "thal": 7}'
python -m cardiosense.clinical.predict --csv patients.csv --output predictions.csv
```

Or run `notebooks/01_clinical_training.ipynb`, which does the same thing with
figures displayed inline.

### ECG (implemented)

```bash
# Full pipeline. GPU. First run builds the ~1 GB waveform cache (several minutes),
# later runs memory-map it instantly.
python -m cardiosense.ecg.train

# Quick smoke test
python -m cardiosense.ecg.train --set training.epochs=2 --skip-baseline --skip-explain

# Experiment E-C: does extra depth earn its cost?
python -m cardiosense.ecg.train --set model.name=resnet1d --experiment-name ecg_resnet
```

If Colab drops mid-run, re-run the same command — it resumes from the next epoch.

Inference:

```bash
python -m cardiosense.ecg.predict --record data/ecg/ptbxl/records100/00000/00001_lr
python -m cardiosense.ecg.predict --npy waveform.npy --output prediction.json
```

Or run `notebooks/02_ecg_training.ipynb`.

### X-ray (implemented)

```bash
# Full pipeline. GPU.
python -m cardiosense.xray.train

# Quick smoke test on a capped subset
python -m cardiosense.xray.train --set dataset.max_images=2000 --set training.epochs=2

# Ablation: train on every negative instead of the 8:1 subsample
python -m cardiosense.xray.train --set dataset.negative_ratio=null
```

Re-run the same command after a Colab disconnect — it resumes, including the
staged-unfreeze state.

Inference:

```bash
python -m cardiosense.xray.predict --image data/xray/nih/images/00000013_005.png
python -m cardiosense.xray.predict --image <path> --gradcam out/cam.png
python -m cardiosense.xray.predict --dir some/folder --output predictions.csv
```

Or run `notebooks/03_xray_training.ipynb`.

Any config value can be overridden from the command line with
`--set dotted.key=value`; values are type-coerced automatically.

---

## 9. Evaluation

Each pipeline writes `results/<modality>/metrics.json` plus figures. Metrics are
chosen per task, never copied blindly between pipelines:

| Pipeline | Headline metric | Why |
|---|---|---|
| Clinical | ROC-AUC + Brier | Near-balanced binary; calibration matters for Phase 2 fusion |
| ECG | Macro ROC-AUC | Multi-label; macro averaging refuses to let the common classes hide the rare ones |
| X-ray | **PR-AUC** | At ~2.5% prevalence, accuracy of 97.5% is achievable by predicting "no" forever |

Every headline number is reported with a bootstrap 95% confidence interval.

---

## 10. Explainability

### Clinical pipeline — four decisions worth defending in the viva

**1. Preprocessing is fit on train only.** The imputer's medians, the encoder's
category levels and the scaler's means are *learned parameters*. Fitting them on
the full dataset before splitting lets test patients shape the training
representation, which inflates every metric invisibly. `tests/test_clinical.py`
asserts that the fitted scaler mean differs from the full-data mean — if it ever
matches, the leak test itself has stopped working.

**2. The threshold is tuned on ~250 points, not 45.** Youden's index computed on
a 45-patient validation split is itself a high-variance estimate. During
development, validation-only tuning picked a 0.65 cut that collapsed test recall
to 0.09; pooling out-of-fold training predictions with validation moved the cut to
0.40 and recall to 0.73 **on identical data**. Both options remain in the config
(`selection.threshold_tuning_data`) so the ablation can be reported.

**3. The calibration method is chosen without circularity.** The final calibrator
is fitted on validation, so the *choice* between Platt and isotonic cannot also be
made there — isotonic would win by fitting it exactly. Instead, out-of-fold
predictions on train give ~200 honest (probability, outcome) pairs, and each
candidate mapping is fitted and scored on **disjoint** subsets of those. An earlier
version of this code got that wrong, picked isotonic, and made the test Brier score
*worse* while emitting calibrated probabilities of exactly 1.0. The current version
picks sigmoid and improves it.

**4. Model selection is not by accuracy, and near-ties go to the simpler model.**
ROC-AUC ranks the candidates; if they land within `tie_tolerance`, Logistic
Regression wins over XGBoost. At this sample size a 0.01 AUC gap is inside the
sampling noise, and the linear model is easier to audit and calibrates more
reliably.

### ECG pipeline — four decisions worth defending in the viva

**1. Multi-label, not multi-class.** A PTB-XL record can carry both `MI` and
`STTC`. Forcing one label per record would delete true positives and corrupt every
metric. This drives the loss (`BCEWithLogitsLoss`), the thresholds (one per class),
and the metric set — `multilabel_metrics` deliberately **refuses to emit
`accuracy`** and reports exact-match ratio under that explicit name instead.

**2. Splits use PTB-XL's official `strat_fold`.** Folds 1–8 / 9 / 10 are
patient-disjoint by the dataset authors' construction, so leakage is impossible
rather than merely avoided — and the numbers stay comparable to published work.
`split_by_fold` asserts zero patient overlap and writes the counts to
`results/ecg/split_summary.json`.

**3. Every preprocessing step is justified individually.** A 0.5 Hz zero-phase
high-pass is applied (drift is not diagnostic and wrecks normalisation); a
low-pass and a mains notch are **not** (they would blunt QRS upstrokes and sit
beyond Nyquist respectively). Per-lead z-scoring is computed within each record,
so it is leakage-free by construction — at the documented cost of discarding
absolute voltage, which matters for `HYP`.

**4. Integrated Gradients is called Integrated Gradients.** Neither architecture
contains attention, so calling the saliency map "attention" would misdescribe the
model. The completeness property (attributions must sum to `F(x) − F(baseline)`)
is checked numerically, and `n_steps` is raised automatically when the check
fails rather than returning a map known not to sum correctly.

### X-ray pipeline — four decisions worth defending in the viva

**1. The split is by patient, and the code asserts it.** A patient contributes
several films; an image-level split puts the same chest on both sides and inflates
AUC by several points while looking entirely normal in the metrics.
`split_by_patient` raises rather than returning a quietly corrupted split, and
negative subsampling drops **whole patients** so it cannot reintroduce the leak.

**2. Horizontal flip is refused, not merely disabled.** It is the default in
almost every ImageNet recipe and it is wrong here: mirroring a chest X-ray moves
the heart to the right side of the thorax, which is dextrocardia. Enabling it in
config raises a `ValueError` naming the reason.

**3. PR-AUC is the headline; accuracy is contextualised.** At the real ~2.5%
prevalence, always-negative scores ~97.5% accuracy and finds nothing. The
majority-class baseline puts exactly that row in the comparison table, and
`evaluate_binary` records `pr_auc_chance_level` (= prevalence) and
`accuracy_of_always_negative` in every metrics block.

**4. One imbalance correction, not several.** `pos_weight` inside
`BCEWithLogitsLoss`. `WeightedRandomSampler` is implemented but disabled: stacking
both applies the correction twice and produces wildly over-confident
probabilities, which would poison the confidence-aware fusion planned for Phase 2.

### Explainability methods

| Pipeline | Method | Why this one |
|---|---|---|
| Clinical | SHAP (TreeExplainer / LinearExplainer) | Exact for tree models; gives both global importance and per-patient attribution |
| ECG | Integrated Gradients (Captum) | Gradient attribution over raw signal; satisfies completeness w.r.t. a baseline |
| X-ray | Grad-CAM on `features.denseblock4` | Standard, faithful to the final convolutional representation |

**These are not proof of medical reasoning.** Saliency shows where gradient mass
concentrated, not why a diagnosis is correct. A Grad-CAM heatmap over the cardiac
silhouette is consistent with the model using heart size; it does not demonstrate
it. This caveat is stated in the results, not buried.

The ECG pipeline uses **Integrated Gradients** and calls it that. It is not
described as attention, because the architecture contains no attention mechanism.

---

## 11. Model artifacts

```text
models/
├── clinical/
│   ├── clinical_model.pkl          final selected estimator
│   ├── clinical_preprocessor.pkl   fitted ColumnTransformer (train-fit only)
│   ├── clinical_calibrator.pkl     fitted calibration wrapper
│   └── clinical_metadata.json      features, classes, split info, metrics, version
├── ecg/
│   ├── ecg_model.pth               best checkpoint weights
│   ├── ecg_config.json             architecture + preprocessing needed for inference
│   └── ecg_metadata.json           label mapping, fold assignment, metrics, version
└── xray/
    ├── xray_model.pth
    ├── xray_config.json
    └── xray_metadata.json          threshold, prevalence, patient-split summary
```

Each metadata file records: label mappings, feature lists, the exact training
configuration, dataset split information, library versions and a model version
string — enough to reload and reproduce inference without guessing.

---

## 12. Project structure

```text
CardioSense/
├── README.md
├── requirements.txt              local / full install
├── requirements-colab.txt        Colab install (no torch — it ships preinstalled)
├── pyproject.toml                editable install, so `import cardiosense` just works
├── .gitignore
│
├── configs/
│   ├── paths.yaml                shared paths, seed, determinism policy
│   ├── clinical_config.yaml
│   ├── ecg_config.yaml
│   └── xray_config.yaml
│
├── data/                         git-ignored; see docs/datasets.md
│   ├── clinical/  ecg/  xray/
│
├── notebooks/
│   ├── 00_colab_setup.ipynb      run this first
│   ├── 01_clinical_training.ipynb
│   ├── 02_ecg_training.ipynb
│   └── 03_xray_training.ipynb
│
├── src/cardiosense/              installable package (src layout)
│   ├── common/                   shared by all three pipelines
│   │   ├── config.py             YAML loading, dotted access, overrides
│   │   ├── paths.py              project-root discovery, no absolute paths
│   │   ├── seeding.py            Python/NumPy/PyTorch seeding + determinism
│   │   ├── logging_utils.py
│   │   ├── env.py                device selection, environment fingerprint
│   │   ├── io_utils.py           NumPy-safe JSON, joblib, CSV
│   │   ├── metrics.py            binary / multiclass / multilabel + calibration
│   │   ├── plots.py              confusion, ROC, PR, calibration, curves
│   │   ├── training.py           checkpoint resume, early stopping, history
│   │   ├── experiment.py         JSON + CSV experiment tracker
│   │   └── compat.py             sklearn 1.6 & torch 2.4 API shims
│   ├── clinical/                 IMPLEMENTED
│   │   ├── data.py               loading, target definition, sentinel cleaning
│   │   ├── eda.py                statistically appropriate EDA (see note below)
│   │   ├── preprocessing.py      stratified split + leakage-free ColumnTransformer
│   │   ├── models.py             LogReg + XGBoost construction and tuning
│   │   ├── evaluate.py           metrics, selection, threshold, error analysis
│   │   ├── calibrate.py          Platt vs isotonic, chosen without leaking
│   │   ├── explain.py            SHAP global + per-patient
│   │   ├── train.py              orchestrator (`python -m cardiosense.clinical.train`)
│   │   └── predict.py            inference (`ClinicalPredictor`)
│   ├── ecg/                      IMPLEMENTED
│   │   ├── data.py               PTB-XL metadata, SCP -> superclass labels, fold splits
│   │   ├── preprocessing.py      filtering, normalisation, waveform cache
│   │   ├── dataset.py            PyTorch Dataset + Colab-tuned DataLoaders
│   │   ├── models.py             1D CNN and ResNet-1D
│   │   ├── baseline.py           statistical features + one-vs-rest LogReg
│   │   ├── trainer.py            resumable training loop, AMP, early stopping
│   │   ├── evaluate.py           multi-label metrics, thresholds, recommendation
│   │   ├── explain.py            Integrated Gradients with completeness check
│   │   ├── train.py              orchestrator (`python -m cardiosense.ecg.train`)
│   │   └── predict.py            inference (`ECGPredictor`)
│   └── xray/                     IMPLEMENTED
│       ├── data.py               NIH metadata, target extraction, patient-level split
│       ├── preprocessing.py      transforms; augmentation guards (no horizontal flip)
│       ├── dataset.py            PNG-reading Dataset + DataLoaders
│       ├── models.py             DenseNet121 + staged unfreezing
│       ├── baseline.py           majority-class + pixel-feature LogReg
│       ├── trainer.py            two-stage fine-tuning, resumable
│       ├── evaluate.py           PR-AUC-led metrics, thresholds, recommendation
│       ├── explain.py            Grad-CAM via hooks
│       ├── train.py              orchestrator (`python -m cardiosense.xray.train`)
│       └── predict.py            inference (`XrayPredictor`)
│
├── scripts/verify_setup.py       one-command environment + dataset check
├── tests/
│   ├── test_common.py            19 fast CPU-only tests
│   ├── test_clinical.py          27 tests incl. explicit leakage checks
│   ├── test_ecg.py               31 tests incl. patient-disjointness and IG completeness
│   └── test_xray.py              32 tests incl. augmentation guards and Grad-CAM hooks
│
├── models/                       artifacts (binaries git-ignored)
├── results/                      metrics, figures, experiment log (committed)
└── docs/
    ├── datasets.md               dataset selection, licensing, download steps
    ├── experiments.md            experiment log and run notes
    └── phase1_results.md         result tables
```

**Deviation from the original spec, and why.** Pipeline code lives in
`src/cardiosense/` rather than directly in `src/`. This makes the project a real
installable package, so notebooks use `from cardiosense.ecg import ...` instead of
`sys.path` manipulation — which is the single most common reason a Colab notebook
runs for one person and not another. The three modalities remain strictly
separated, and a `common/` package holds only genuinely shared utilities.

---

## 13. Limitations

Stated up front, because a decision-support project that overstates itself is
worse than one that underperforms:

- **The clinical dataset has 303 patients.** The test split is ~45 people. All
  metrics carry wide confidence intervals, which are reported rather than hidden.
- **ChestX-ray14 labels are NLP-mined from reports**, with a literature-estimated
  error rate near 10%. Measured performance is bounded by label noise.
- **Explainability shows correlation with model output, not clinical reasoning.**
  SHAP, Integrated Gradients and Grad-CAM are plausibility checks.
- **No external validation.** None of these cohorts is Indian; performance will
  not transfer without local validation.
- **Negative subsampling in the X-ray pipeline** changes the prevalence, so
  emitted probabilities are conditioned on the sampled prior. The sampling ratio
  is recorded in the model metadata for later prior correction.
- **This is a research artifact, not a medical device.** No output should be used
  for clinical decision-making.

---

## 14. Phase 2 (future work)

1. **Confidence-aware multimodal fusion** — weight each modality by its
   calibrated reliability, not just its point prediction. Phase 1's calibration
   work exists to make this possible.
2. **Guideline-grounded RAG** over ACC/AHA and ESC cardiovascular guidelines.
3. **LLM-based recommendation and clinical report generation**, constrained to
   retrieved guideline text.
4. **FastAPI backend + React dashboard.**
5. **CheXpert replication**, using its uncertainty labels properly.
6. **External validation** on a local cohort.

---

## 15. Reproducibility

- `SEED = 42` seeds Python, NumPy and PyTorch in every entry point.
- Strict determinism is configurable per pipeline (`strict_determinism` in
  `configs/paths.yaml`). It is **on** for the clinical pipeline (free) and **off**
  for the CNNs, where deterministic cuDNN kernels cost roughly 10–30% throughput.
  Instead of paying that, ECG and X-ray results are reported as mean ± std across
  three seeds.
- Every run writes a JSON record with the full config, git commit, library
  versions and hardware to `results/experiments/`.
- `pytest` runs a CPU-only smoke suite that needs no datasets.

---

## 16. Licence and citation

Code: MIT. Datasets retain their own licences (PTB-XL: CC-BY 4.0; ChestX-ray14:
NIH terms of use; UCI Heart Disease: UCI ML Repository terms). Cite the original
dataset papers, listed in [`docs/datasets.md`](docs/datasets.md), in any
publication.
