# Datasets — selection rationale and acquisition

Every dataset below is real, public and cited. Nothing in CardioSense is
synthetic, and no label is invented: each target is derived from a field that
already exists in the source data.

---

## 1. Clinical — UCI Heart Disease (Cleveland)

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset ID **45** ("Heart Disease") |
| Access | Open. Downloaded automatically by `ucimlrepo` — no manual step |
| Records | 303 patients (Cleveland subset) |
| Features | 13 predictors |
| Raw target | `num` — angiographic disease severity, 0–4 |
| Phase 1 target | **Binary**: `num == 0` → 0, `num > 0` → 1 |
| Positive rate | ≈ 46% (near-balanced) |
| Citation | Janosi, Steinbrunn, Pfisterer & Detrano (1989), *Heart Disease*, UCI ML Repository |

### Why this dataset

It is the reference benchmark for tabular cardiovascular risk prediction, the
features are all routinely collected clinical variables (age, sex, chest pain
type, resting BP, cholesterol, fasting glucose, resting ECG, max heart rate,
exercise angina, ST depression, ST slope, fluoroscopy vessel count, thalassemia
scan), and the label is angiographically confirmed rather than self-reported.
For Phase 2 this matters: every feature is something that would plausibly be
available alongside an ECG and a chest film for the same patient.

### The honest caveat, and what we do about it

**303 rows is small.** With a 70/15/15 split the test set is ~45 patients, so a
single misclassification moves accuracy by 2.2 points and the ROC-AUC confidence
interval is roughly ±0.10. This is stated plainly in the results rather than
hidden. Two mitigations are built into the pipeline:

1. Every headline metric is reported with a **bootstrap 95% CI**, not as a bare
   point estimate.
2. Model selection and hyperparameter search use **stratified 5-fold CV on the
   training split**, never a single validation draw.

### Optional larger variant

The same UCI donation includes three further sites — Hungarian (294),
Switzerland (123) and Long Beach VA (200) — for a pooled **920 patients**.
Pooling roughly triples the data, at the cost of much heavier missingness
(Switzerland has `chol` recorded as 0 for most patients, which is a missing-data
sentinel, not a measurement).

CardioSense keeps **Cleveland as the primary dataset** because it is complete,
comparable to published baselines and adequate for the calibration study. The
pooled version is supported as a documented ablation: place the combined CSV at
`data/clinical/heart_disease.csv` and set

```yaml
dataset:
  source: local_csv
```

in `configs/clinical_config.yaml`. If you do this, treat `chol == 0` and
`trestbps == 0` as missing before imputation — otherwise the model learns the
site, not the disease.

### Target definition (state this in the report)

> A patient is labelled positive when angiography showed >50% diameter narrowing
> in at least one major coronary vessel (`num > 0`). Severity grading (1–4) is
> collapsed to a binary presence label; graded severity prediction is out of
> Phase 1 scope.

---

## 2. ECG — PTB-XL

| | |
|---|---|
| Source | PhysioNet, `ptb-xl` v1.0.3 |
| Access | **Open access** — no credentialing, no data-use agreement form |
| Licence | Creative Commons Attribution 4.0 |
| Records | ~21,800 clinical 12-lead ECGs from ~18,900 patients |
| Duration | 10 s per record |
| Sampling | Provided at **both** 100 Hz (`records100/`) and 500 Hz (`records500/`) |
| Leads | I, II, III, aVR, aVL, aVF, V1–V6 |
| Format | WFDB (`.dat` + `.hea`), read with the `wfdb` package |
| Citation | Wagner et al. (2020), *PTB-XL, a large publicly available ECG dataset*, Scientific Data |

### Files that matter

```
ptbxl/
├── ptbxl_database.csv     # one row per record: patient_id, filename_lr,
│                          # filename_hr, scp_codes, strat_fold, age, sex, ...
├── scp_statements.csv     # SCP-ECG code -> diagnostic / form / rhythm class
├── records100/            # 100 Hz waveforms, nested 00000/ 01000/ ... folders
└── records500/            # 500 Hz waveforms, same structure
```

`scp_codes` is a **stringified Python dict** (e.g. `{'NORM': 100.0, 'SR': 0.0}`)
mapping SCP code to a likelihood in 0–100. A likelihood of `0.0` means the
likelihood was not stated, **not** that the statement is absent — it is treated
as present, following the convention used in the PTB-XL benchmarking paper.

### Selected task: 5-class diagnostic superclass, multi-label

Joining `scp_codes` against the `diagnostic_class` column of
`scp_statements.csv` collapses 71 SCP statements into five superclasses:

| Class | Meaning | Approx. share of records |
|---|---|---|
| `NORM` | Normal ECG | ~44% |
| `MI` | Myocardial infarction | ~25% |
| `STTC` | ST/T change | ~24% |
| `CD` | Conduction disturbance | ~22% |
| `HYP` | Hypertrophy | ~12% |

(Shares sum to more than 100% because records carry multiple superclasses.)

**Why this task and not another:**

- **Clinically meaningful.** Each superclass maps to a real decision point, and
  MI/STTC/HYP are precisely the findings that a cardiovascular risk model would
  want to know about — which makes this the right output to feed into the Phase 2
  fusion stage.
- **Sufficiently represented.** The rarest class still has ~2,600 records.
  Attempting all 71 SCP statements would leave dozens of classes with fewer than
  50 examples, where nothing can be learned and no metric is trustworthy.
- **Suitable for deep learning.** ~17,000 training records is enough for a
  modest 1D CNN without pretraining.
- **Evaluable.** It is the standard PTB-XL benchmark task, so our macro-AUC is
  directly comparable to published numbers instead of being unreviewable.

**Multi-label, not multi-class.** A record genuinely can be both `MI` and `STTC`.
Forcing one label per record would silently delete true positives and make every
reported metric wrong. The pipeline therefore uses `BCEWithLogitsLoss`, per-class
thresholds, and multi-label metrics — and does **not** report plain accuracy,
which is undefined here. Exact-match ratio is reported instead, under that name.

### Splitting

PTB-XL ships a `strat_fold` column (1–10): a stratified, **patient-disjoint**
assignment produced by the dataset authors. CardioSense uses folds **1–8 train,
9 validation, 10 test**. This is the published convention and it makes
leakage impossible by construction. **Never re-split PTB-XL randomly by record** —
patients contribute multiple ECGs, and a random record-level split puts the same
patient on both sides.

### Why 100 Hz

The full training set at 100 Hz is about 1.0 GB as float32 (21.8k × 12 × 1000 × 4 B),
which fits in Colab RAM and can be memory-mapped from a cached `.npy`. At 500 Hz
it is 5 GB and every epoch becomes I/O-bound. The morphology that distinguishes
these five superclasses lives well below 50 Hz, so 100 Hz loses nothing relevant.
The sampling rate is a config switch, so a 500 Hz ablation is a one-line change.

### Download

Open access, so a single command works:

```bash
# ~1.7 GB zip
wget -q --show-progress \
  https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip \
  -O ptbxl.zip
unzip -q ptbxl.zip
mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 $CARDIOSENSE_DATA_ROOT/ecg/ptbxl
```

Verify with:

```bash
ls $CARDIOSENSE_DATA_ROOT/ecg/ptbxl
# expect: ptbxl_database.csv  scp_statements.csv  records100/  records500/  ...
```

If you only want the 100 Hz waveforms, you can delete `records500/` afterwards and
save ~1.2 GB.

---

## 3. Chest X-ray — NIH ChestX-ray14 (chosen over CheXpert)

| | |
|---|---|
| Source | NIH Clinical Center, ChestX-ray14 |
| Access | Open download (Kaggle mirror `nih-chest-xrays/data`, or NIH Box links) |
| Images | 112,120 frontal-view PNGs, 1024×1024, from 30,805 patients |
| Labels | 14 findings, NLP-mined from radiology reports |
| Phase 1 target | **Cardiomegaly**, binary |
| Prevalence | ≈ 2.5% of all images (heavily imbalanced) |
| Citation | Wang et al. (2017), *ChestX-ray8/14*, CVPR |

### Why ChestX-ray14 rather than CheXpert

| | ChestX-ray14 | CheXpert |
|---|---|---|
| Access | Immediate, open | Stanford AIMI registration + signed agreement; approval is not instant |
| Metadata | One flat CSV | Separate train/valid CSVs, path-encoded metadata |
| Patient IDs | Explicit `Patient ID` column | Encoded in the file path |
| Label semantics | Binary present/absent | Four states: positive, negative, **uncertain (-1)**, blank |
| Size | ~45 GB | ~440 GB (full res) |

The deciding factor is **practicality within a B.Tech timeline**. CheXpert's
uncertainty labels are genuinely interesting — they are arguably better suited to
the confidence-aware theme of this project — but they force an uncertainty policy
decision (U-Ones vs U-Zeros vs U-Ignore) that is a research contribution in
itself, and the access wait can burn a fortnight. ChestX-ray14 gets a working,
honestly-evaluated pipeline finished. **A CheXpert replication is listed as
Phase 2 future work**, where the uncertainty labels can be used properly.

### Target and filtering

- **Positive** iff the string `Cardiomegaly` appears in the pipe-separated
  `Finding Labels` column. No label is created or inferred.
- **PA views only** (`View Position == "PA"`). AP films are taken with portable
  equipment on supine, sicker patients and geometrically magnify the cardiac
  silhouette. Mixing views hands the model a shortcut — predict cardiomegaly
  whenever the film looks like a portable AP — that has nothing to do with heart
  size.
- **Negative subsampling** (`negative_ratio: 8`) keeps a Colab epoch to a few
  minutes. It changes the *prevalence*, so every probability the model emits is
  conditioned on the sampled prevalence; the evaluation code records the sampling
  ratio in the metadata so probabilities can be prior-corrected in Phase 2.
  Set `negative_ratio: null` to train on everything.

### Splitting — patient level, non-negotiable

A patient can contribute dozens of films. An image-level split puts the same
chest in train and test and inflates AUC by several points; this is one of the
best-documented failure modes in chest X-ray literature. CardioSense uses NIH's
own **patient-disjoint** `train_val_list.txt` / `test_list.txt`, and carves
validation out of the train-val list **by patient**. The split code asserts that
the patient-ID intersection between every pair of splits is empty and writes the
counts to `results/xray/split_summary.json`.

### Why accuracy is not the metric

At ~2.5% prevalence, "no cardiomegaly, always" scores ~97.5% accuracy and is
worthless. The headline metric is **PR-AUC**, whose chance level equals the
positive prevalence; ROC-AUC is reported for comparability with the literature,
and the operating threshold is chosen on validation, never on test.

### Download

The full set is ~45 GB, which will not fit in a free 15 GB Google Drive. The
workflow used here:

1. Download to the **Colab local disk** (`/content`, ~100 GB, ephemeral).
2. Build the filtered PA-Cardiomegaly subset once (~10k images, ~1.5 GB).
3. Copy **only that subset** to Drive, so later sessions skip the 45 GB download.

```bash
# 1. Kaggle credentials: Kaggle > Settings > API > Create New Token
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Download + unzip (~45 GB, 20-40 min on Colab)
kaggle datasets download -d nih-chest-xrays/data -p /content/nih_raw
unzip -q /content/nih_raw/data.zip -d /content/nih_raw
```

**Layout gotcha.** The Kaggle mirror ships images in twelve folders
(`images_001/images/`, …, `images_012/images/`) and names the label file
`Data_Entry_2017.csv`, whereas the NIH v2020 release calls it
`Data_Entry_2017_v2020.csv`. Flatten and normalise once:

```bash
DEST=$CARDIOSENSE_DATA_ROOT/xray/nih
mkdir -p $DEST/images
find /content/nih_raw -path '*/images/*.png' -exec mv -t $DEST/images {} +
cp /content/nih_raw/Data_Entry_2017*.csv $DEST/
cp /content/nih_raw/train_val_list.txt /content/nih_raw/test_list.txt $DEST/
# make the filename match the config (or edit configs/xray_config.yaml instead)
[ -f $DEST/Data_Entry_2017.csv ] && \
  mv $DEST/Data_Entry_2017.csv $DEST/Data_Entry_2017_v2020.csv
```

Verify with:

```bash
ls $DEST                      # Data_Entry_2017_v2020.csv images/ test_list.txt train_val_list.txt
ls $DEST/images | wc -l       # expect 112120
```

---

## Storage summary

| Dataset | Raw size | Size after Phase 1 filtering | Where it should live |
|---|---|---|---|
| UCI Heart Disease | < 100 KB | < 100 KB | Repo-adjacent `data/clinical/` |
| PTB-XL (100 Hz only) | ~500 MB | ~1.0 GB `.npy` cache | Google Drive |
| ChestX-ray14 | ~45 GB | ~1.5 GB subset | Colab disk → subset to Drive |

Set the data root once per session so nothing needs editing:

```python
import os
os.environ["CARDIOSENSE_DATA_ROOT"] = "/content/drive/MyDrive/CardioSense/data"
```

---

## Ethics and intended use

All three datasets are de-identified and publicly released for research. The
models built in Phase 1 are **research artifacts for a final-year project**, not
medical devices, and must not be used for clinical decision-making. Two specific
limitations to carry into the report:

- ChestX-ray14 labels are **NLP-mined from free-text reports**, with an estimated
  label error rate around 10%. The ceiling on measurable performance is set by
  label noise, not only by the model.
- None of these cohorts is Indian. Prevalence, body habitus and referral patterns
  differ, so measured performance does not transfer to a local population without
  external validation.
