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
