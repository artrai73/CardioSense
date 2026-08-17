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

_(Populated as each pipeline is built. The interface is fixed now so the
notebooks and scripts stay in step.)_

```bash
# Clinical  — CPU, ~2 minutes
python -m cardiosense.clinical.train  --config configs/clinical_config.yaml

# ECG       — GPU, ~25 minutes for 40 epochs at 100 Hz
python -m cardiosense.ecg.train       --config configs/ecg_config.yaml

# X-ray     — GPU, ~45 minutes for 15 epochs on the filtered subset
python -m cardiosense.xray.train      --config configs/xray_config.yaml
```

Any config value can be overridden without editing YAML:

```bash
python -m cardiosense.ecg.train --config configs/ecg_config.yaml \
    --set training.epochs=2 --set model.name=resnet1d
```

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
│   ├── clinical/                 preprocessing, train, evaluate, calibrate, explain, predict
│   ├── ecg/                      preprocessing, dataset, models, train, evaluate, explain, predict
│   └── xray/                     preprocessing, dataset, models, train, evaluate, explain, predict
│
├── scripts/verify_setup.py       one-command environment + dataset check
├── tests/test_common.py          fast CPU-only smoke tests
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
