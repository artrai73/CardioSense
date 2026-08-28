"""UCI Heart Disease loading, target definition and cleaning.

This module is the ONLY place that knows how the raw dataset is shaped. Everything
downstream receives a clean ``(X, y)`` pair plus a report describing exactly what
was changed, so the cleaning steps end up in the results rather than in someone's
memory.

Target definition (state this verbatim in the report):

    A patient is positive when angiography showed >50% diameter narrowing in at
    least one major coronary vessel, i.e. the raw severity label ``num > 0``.
    Severity grading 1-4 is collapsed to binary presence; graded prediction is
    out of Phase 1 scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.paths import PATHS

__all__ = [
    "FEATURE_DESCRIPTIONS",
    "CATEGORICAL_LEVELS",
    "load_raw_dataframe",
    "prepare_dataset",
    "describe_features",
]

logger = get_logger(__name__)

#: Human-readable meaning of every column. Used in EDA titles and SHAP plots so
#: that a reader does not have to decode ``thal`` from memory.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg on admission)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium scan (3 normal, 6 fixed defect, 7 reversible defect)",
    "num": "Angiographic disease severity (0-4); binarised to presence/absence",
}

#: Readable level names for categorical codes, used on EDA axis labels.
CATEGORICAL_LEVELS: dict[str, dict[float, str]] = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T abnormal", 2: "LV hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}

#: Columns where a recorded 0 is physiologically impossible and therefore a
#: missing-data sentinel, not a measurement. Cleveland has none of these; the
#: pooled 4-hospital variant has many (Switzerland records chol = 0 throughout).
_ZERO_IS_MISSING = ("chol", "trestbps")


def _resolve_data_path(value: str | Path) -> Path:
    """Resolve a config path, honouring ``$CARDIOSENSE_DATA_ROOT`` for ``data/...``."""
    text = str(value)
    if text.startswith("data/"):
        return (PATHS.data / text[len("data/"):]).resolve()
    path = Path(text).expanduser()
    return path if path.is_absolute() else (PATHS.root / path).resolve()


def load_raw_dataframe(cfg: Config, force_download: bool = False) -> pd.DataFrame:
    """Load the raw dataset as a DataFrame, caching the download.

    Two sources, chosen by ``cfg.dataset.source``:

    * ``uci_api``   — fetched with ``ucimlrepo`` (dataset id 45) and cached to
      ``cfg.dataset.raw_cache_path`` so later runs work offline.
    * ``local_csv`` — read from ``cfg.dataset.local_csv_path``. Use this for the
      pooled 920-patient variant, or when the runtime has no internet.

    Args:
        cfg: Clinical configuration.
        force_download: Ignore the cache and refetch.

    Returns:
        The raw DataFrame, features and target together, exactly as distributed.

    Raises:
        FileNotFoundError: If ``local_csv`` is selected and the file is absent.
        RuntimeError: If the download fails and no cache exists.
    """
    source = cfg.dataset.source
    cache_path = _resolve_data_path(cfg.dataset.raw_cache_path)

    if source == "local_csv":
        csv_path = _resolve_data_path(cfg.dataset.local_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"local_csv source selected but {csv_path} does not exist.\n"
                "Either place the CSV there or set dataset.source: uci_api."
            )
        logger.info("Loading clinical data from local CSV: %s", csv_path)
        return pd.read_csv(csv_path)

    if source != "uci_api":
        raise ValueError(f"Unknown dataset.source {source!r}; expected 'uci_api' or 'local_csv'.")

    if cache_path.exists() and not force_download:
        logger.info("Loading cached UCI download: %s", cache_path)
        return pd.read_csv(cache_path)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "ucimlrepo is not installed and no cached copy exists.\n"
            "Run: pip install ucimlrepo"
        ) from exc

    repo_id = int(cfg.dataset.uci_repo_id)
    logger.info("Downloading UCI dataset id=%d ...", repo_id)
    try:
        repo = fetch_ucirepo(id=repo_id)
    except Exception as exc:  # noqa: BLE001 - surface any network/API failure clearly
        raise RuntimeError(
            f"Could not download UCI dataset id={repo_id}: {exc}\n"
            "If this runtime has no internet, download the CSV elsewhere, place it at "
            f"{_resolve_data_path(cfg.dataset.local_csv_path)} and set dataset.source: local_csv."
        ) from exc

    frame = pd.concat([repo.data.features, repo.data.targets], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    logger.info("Downloaded %d rows x %d cols, cached to %s",
                frame.shape[0], frame.shape[1], cache_path)
    return frame


def prepare_dataset(
    frame: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Clean the raw frame and split it into features and a binary target.

    Steps, all recorded in the returned report:

    1. Verify every configured column is present.
    2. Replace physiologically impossible zeros in ``chol`` / ``trestbps`` with
       ``NaN`` (missing-data sentinels, not measurements).
    3. Drop exact duplicate rows — the same patient recorded twice would leak
       across the train/test split.
    4. Drop rows with a missing target (a row with no label teaches nothing and
       cannot be scored).
    5. Binarise the target when ``dataset.binarize_target`` is set.

    Missing *feature* values are deliberately NOT imputed here. Imputation is a
    fitted transformation and belongs inside the ColumnTransformer, which is fit
    on the training split only. Imputing here would leak test statistics into
    training.

    Args:
        frame: Raw DataFrame from :func:`load_raw_dataframe`.
        cfg: Clinical configuration.

    Returns:
        ``(X, y, report)``.

    Raises:
        KeyError: If a configured column is missing from the data.
        ValueError: If the target ends up with fewer than two classes.
    """
    df = frame.copy()
    report: dict[str, Any] = {"rows_in": int(len(df)), "columns_in": list(df.columns)}

    target_col = cfg.dataset.target_column
    numeric = list(cfg.dataset.numeric_features)
    categorical = list(cfg.dataset.categorical_features)
    drop = list(cfg.dataset.get("drop_features", []) or [])

    missing_cols = [c for c in (*numeric, *categorical, target_col) if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Columns missing from the dataset: {missing_cols}\n"
            f"Available columns: {list(df.columns)}\n"
            "If you switched to the pooled 4-hospital CSV, check its header row."
        )

    # -- 2. sentinel zeros -> NaN ------------------------------------------
    sentinel_counts: dict[str, int] = {}
    for column in _ZERO_IS_MISSING:
        if column in df.columns:
            n_zero = int((df[column] == 0).sum())
            if n_zero:
                df.loc[df[column] == 0, column] = np.nan
                sentinel_counts[column] = n_zero
                logger.warning(
                    "%s: %d rows had a value of 0, which is physiologically impossible. "
                    "Treated as missing.", column, n_zero,
                )
    report["sentinel_zeros_to_nan"] = sentinel_counts

    # -- 3. duplicates ------------------------------------------------------
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.warning("Dropped %d exact duplicate rows (they would leak across splits).",
                       n_duplicates)
    report["duplicates_dropped"] = n_duplicates

    # -- 4. rows without a label -------------------------------------------
    n_missing_target = int(df[target_col].isna().sum())
    if n_missing_target:
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        logger.warning("Dropped %d rows with a missing target.", n_missing_target)
    report["rows_missing_target_dropped"] = n_missing_target

    # -- 5. target ----------------------------------------------------------
    raw_target = df[target_col]
    report["raw_target_distribution"] = {
        str(k): int(v) for k, v in raw_target.value_counts().sort_index().items()
    }

    if bool(cfg.dataset.get("binarize_target", True)):
        y = (raw_target.astype(float) > 0).astype(int)
        report["target_rule"] = f"{target_col} > 0 -> 1 (disease present), else 0"
    else:
        y = raw_target.astype(int)
        report["target_rule"] = f"{target_col} used as-is"
    y.name = cfg.dataset.get("target_name", "target")

    if y.nunique() < 2:
        raise ValueError(
            f"Target has only one class after preparation: {y.unique()}. "
            "Check dataset.target_column and dataset.binarize_target."
        )

    feature_columns = [c for c in (*numeric, *categorical) if c not in drop]
    X = df[feature_columns].copy()

    # Categorical codes arrive as floats (because NaN forces float dtype). Keep
    # them numeric here; the OneHotEncoder in the preprocessor handles the
    # conversion, and imputation happens there too.
    for column in categorical:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")

    report.update({
        "rows_out": int(len(X)),
        "n_features": int(X.shape[1]),
        "numeric_features": [c for c in numeric if c in X.columns],
        "categorical_features": [c for c in categorical if c in X.columns],
        "target_name": str(y.name),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "positive_rate": float(y.mean()),
        "missing_values_per_feature": {
            c: int(n) for c, n in X.isna().sum().items() if n > 0
        },
    })

    logger.info(
        "Prepared dataset: %d rows, %d features, positive rate %.1f%%",
        report["rows_out"], report["n_features"], 100 * report["positive_rate"],
    )
    return X, y, report


def describe_features(columns: list[str]) -> pd.DataFrame:
    """Return a tidy description table for the given columns (used in EDA)."""
    return pd.DataFrame(
        {"feature": columns,
         "description": [FEATURE_DESCRIPTIONS.get(c, "—") for c in columns]}
    )
