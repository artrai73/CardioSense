# Experiment log

Human-readable companion to the machine-written log in
`results/experiments/experiment_log.csv` (one row per run) and
`results/experiments/<run_id>.json` (full record: config, params, metrics,
history, environment, git commit).

**Rule: every number in this file must be traceable to a `run_id`.**
Nothing gets written here by hand before the run exists.

## How a run is recorded

```python
from cardiosense.common import ExperimentTracker, load_config

cfg = load_config("clinical")
with ExperimentTracker("xgboost_tuned", modality="clinical",
                       config=cfg, primary_metric="roc_auc") as run:
    run.log_params({"model": "xgboost", "n_estimators": 300, "max_depth": 3})
    run.log_metrics(val_metrics, split="val")
    run.log_metrics(test_metrics, split="test")
    run.log_artifact("model", model_path)
```

Captured automatically: timestamp, duration, status, seed, full config, git
commit, Python/library versions, CPU count, GPU model and CUDA version.

## Planned Phase 1 experiments

| ID | Modality | Experiment | Purpose |
|---|---|---|---|
| C-A | Clinical | Logistic Regression | Interpretable baseline |
| C-B | Clinical | XGBoost (RandomizedSearchCV) | Does non-linearity help on 303 rows? |
| C-C | Clinical | Calibrated XGBoost (Platt vs isotonic) | Reliability of the probability itself |
| E-A | ECG | Statistical features + one-vs-rest LogReg | Floor for the CNN to clear |
| E-B | ECG | 1D CNN | Main model |
| E-C | ECG | ResNet-1D | Run only if E-B is stable and time allows |
| X-A | X-ray | Majority class + pixel-feature LogReg | Shows why accuracy is useless here |
| X-B | X-ray | DenseNet121 transfer learning | Main model |

Cross-cutting comparison at the end of Phase 1: predictive performance,
computational cost (training minutes, parameter count, inference latency),
reliability (Brier, ECE), and explainability (what SHAP / IG / Grad-CAM actually
show, and what they do not).

## Run notes

_(Appended as runs complete. Format: run_id — what changed — what happened — what it means.)_

| Date | run_id | Change | Result | Interpretation |
|---|---|---|---|---|
| | | | | |

## Clinical pipeline — build notes

Two methodological bugs were found and fixed during construction. Both are worth
writing up, because both are the kind that produce plausible-looking numbers
rather than crashes.

**Threshold tuned on too little data.** Youden's index computed on the
45-patient validation split alone selected a 0.65 cut, which collapsed test
recall to 0.09. Pooling out-of-fold training predictions with validation
(~250 points) moved the cut to 0.40 and recall to 0.73 on identical data.
Controlled by `selection.threshold_tuning_data`; both settings are kept so the
comparison can be reported as an ablation.

**Circular calibration-method selection.** The first implementation fitted each
candidate calibrator on a CV fold and scored it on *that same fold*. Isotonic
regression wins that comparison automatically — being non-parametric, it can fit
the fold almost exactly. It was duly selected, then made the test Brier score
worse (0.225 -> 0.231) and emitted calibrated probabilities of exactly 1.0 at
inference. The fix scores each candidate on held-out pairs drawn from
out-of-fold base predictions; sigmoid then wins honestly and improves test Brier
(0.225 -> 0.193) and ECE (0.220 -> 0.134).

Calibrated probabilities are additionally clipped to [0.001, 0.999]: a stated
certainty of 1.0 is not something a few hundred patients can support.


## ECG pipeline — build notes

Four issues found by actually running the pipeline. All four produced plausible
output rather than crashing, which is what makes them worth recording.

**Read-only memmap views reaching PyTorch.** The waveform cache is memory-mapped,
and indexing a mmap returns a *read-only* view. `torch.from_numpy` on a read-only
buffer warns and yields a tensor PyTorch treats as unsafe to write, and handing a
mmap view to a DataLoader worker process is undefined behaviour. Fixed by copying
explicitly in `ECGDataset.__getitem__` (48 KB per record at 100 Hz — negligible).
The test `test_dataset_returns_writable_tensors` writes into the returned tensor,
which would fail on a read-only buffer.

**Best-epoch off-by-one.** `CheckpointManager` stores the 0-based loop index
(which is what resume arithmetic needs), and this was being reported directly, so
a model whose best epoch was the first was logged as "epoch 0". Now converted once
in `train_model`; resume still uses the 0-based value internally.

**Integrated Gradients completeness check was misleading.** The relative
convergence error divides by `F(x) - F(baseline)`. When the model is nearly
indifferent about a record that denominator approaches zero and the relative error
explodes — the first run reported errors up to 3.2 on attributions that were
numerically fine. Now both absolute and relative error are computed and a failure
is declared only when both exceed tolerance; on failure, `n_steps` doubles (up to
1024) and the attribution is recomputed. After the fix every explanation converged
with absolute error under 0.008.

**The deep model did not automatically beat the baseline.** On the scratch
validation data the statistical baseline scored a higher macro AUC than the
undertrained CNN. Rather than leave that for a human to notice, `recommend_model`
now makes the comparison explicit, applies a 0.01 macro-AUC bar (below which
seed-to-seed variation explains the difference), and emits a warning when the
classical baseline wins. This is the check the brief asks for regarding ResNet-1D,
generalised to every model in the comparison.

### Guidance for Experiment E-C (ResNet-1D)

Only run it once the plain CNN trains stably — a stable, converging CNN is the
control. Then:

```bash
python -m cardiosense.ecg.train --set model.name=resnet1d --experiment-name ecg_resnet
```

`recommend_model` will report whether the extra depth cleared the 0.01 bar. If it
did not, keep the CNN and say so in the paper; a 0.003 macro-AUC gain on a ~4x
larger model is not a finding, it is noise.


## X-ray pipeline — build notes

**A shared checkpoint bug, found by the X-ray resume test.** `CheckpointManager`
wrote `last.pt` *before* updating its best-value tracker, so `last.pt` always held
the **previous** epoch's best. Resuming therefore restored a stale, worse baseline:
early stopping restarted its patience count from the wrong value, and an inferior
checkpoint could overwrite `best.pt`. It was visible in the resume log as
"Resumed ... best val_pr_auc = 0.4519" when the actual best was 0.6917.

The check now happens before either file is written. This affected **both** the
ECG and X-ray pipelines, since they share `common/training.py`. Two regression
tests cover it: `test_last_checkpoint_records_the_current_best` and
`test_resume_restores_the_correct_best_value`, the second of which asserts that a
worse epoch after resuming does not overwrite `best.pt`.

**Augmentation guards are errors, not comments.** Enabling `horizontal_flip` or
`vertical_flip` in the config raises a `ValueError` naming the reason. A comment
saying "don't turn this on" is not a safeguard; someone copying a standard
ImageNet recipe would flip it on without reading.

**Grad-CAM hook cleanup is tested.** `GradCAM` is a context manager, and
`test_gradcam_removes_its_hooks` asserts the hook count returns to its original
value. A leaked backward hook silently slows every later forward pass and holds
tensors alive — a failure that shows up as mysterious slowdown, not as an error.

**Prevalence spread across patient-level splits.** Patient-level splitting cannot
stratify perfectly, because patients contribute different numbers of images. The
split code warns when the prevalence spread exceeds 10 points, because a PR-AUC
difference between splits then partly reflects different chance levels rather than
model quality.

### What to check on the real dataset

1. **Does DenseNet121 beat the pixel-feature baseline by more than 0.02 PR-AUC?**
   `recommend_model` answers this automatically and warns if not. A logistic
   regression on a 32x32 thumbnail matching the CNN would mean the task is being
   solved by global exposure and body size, not cardiac silhouette shape.
2. **Do the Grad-CAM maps sit on the heart?** `mass_lower_half` and
   `mass_central_third` in `gradcam_summary.json` are crude numeric checks. They
   are weak evidence, not validation.
3. **What co-occurs with the false positives?** `false_positive_findings` in the
   error summary. A single finding dominating that list suggests the model is
   keying on it rather than on heart size.
