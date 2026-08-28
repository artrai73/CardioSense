"""Exploratory data analysis for the clinical dataset.

Everything here writes to ``results/clinical/eda/`` and returns a JSON-able
summary, so the EDA is reproducible rather than a scroll of notebook output.

**On not plotting inappropriate statistics.** The dataset mixes genuinely
continuous variables with integer-coded categoricals. A Pearson correlation
matrix over all 13 columns would put ``thal`` (3 = normal, 6 = fixed defect,
7 = reversible defect) on the same footing as ``age``, implying an ordering and a
distance that do not exist. So:

* Pearson correlation is computed for **numeric features only**.
* Association between a **categorical feature and the target** uses Cramer's V,
  which is defined for nominal variables.
* Association between a **numeric feature and the binary target** uses the
  point-biserial correlation, which is the appropriate special case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..common.config import Config
from ..common.io_utils import save_dataframe, save_json
from ..common.logging_utils import get_logger
from ..common.paths import ensure_dir
from ..common.plots import plot_class_distribution, save_figure
from .data import CATEGORICAL_LEVELS, FEATURE_DESCRIPTIONS

__all__ = ["run_eda", "cramers_v"]

logger = get_logger(__name__)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V association between two categorical variables (0 = none, 1 = perfect).

    Uses the bias correction of Bergsma (2013), which matters here because some
    categories (``thal`` = fixed defect) have few observations.
    """
    table = pd.crosstab(x, y)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else float("nan")


def _label(column: str, value: Any) -> str:
    """Map a categorical code to a readable level name where one is known."""
    levels = CATEGORICAL_LEVELS.get(column, {})
    try:
        return levels.get(float(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def run_eda(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Run the full EDA suite and save every figure and table.

    Args:
        X: Feature frame (uncleaned of missing values — missingness is a finding).
        y: Binary target.
        cfg: Clinical configuration.
        out_dir: Directory for figures and tables, e.g. ``results/clinical/eda``.

    Returns:
        A JSON-serialisable summary, also written to ``eda_summary.json``.
    """
    import matplotlib.pyplot as plt

    out = ensure_dir(out_dir)
    numeric = [c for c in cfg.dataset.numeric_features if c in X.columns]
    categorical = [c for c in cfg.dataset.categorical_features if c in X.columns]
    target_name = str(y.name)

    summary: dict[str, Any] = {
        "shape": {"rows": int(X.shape[0]), "features": int(X.shape[1])},
        "target_name": target_name,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }

    # -- 1. dtypes, missingness, duplicates ---------------------------------
    overview = pd.DataFrame({
        "feature": X.columns,
        "role": ["numeric" if c in numeric else "categorical" for c in X.columns],
        "dtype": [str(X[c].dtype) for c in X.columns],
        "n_unique": [int(X[c].nunique(dropna=True)) for c in X.columns],
        "n_missing": [int(X[c].isna().sum()) for c in X.columns],
        "pct_missing": [round(100 * X[c].isna().mean(), 2) for c in X.columns],
        "description": [FEATURE_DESCRIPTIONS.get(c, "—") for c in X.columns],
    })
    save_dataframe(overview, out / "feature_overview.csv")
    summary["feature_overview"] = overview.to_dict(orient="records")
    summary["duplicate_rows"] = int(X.duplicated().sum())
    summary["total_missing_cells"] = int(X.isna().sum().sum())

    # -- 2. class distribution ---------------------------------------------
    counts = {f"{target_name}={int(k)}": int(v) for k, v in y.value_counts().sort_index().items()}
    plot_class_distribution(counts, out / "class_distribution.png",
                            title=f"Target distribution ({target_name})")
    summary["class_distribution"] = counts
    summary["positive_rate"] = float(y.mean())
    summary["imbalance_ratio"] = float(max(counts.values()) / max(min(counts.values()), 1))

    # -- 3. descriptive statistics -----------------------------------------
    described = X[numeric].describe().T
    described["missing"] = X[numeric].isna().sum()
    described["skew"] = X[numeric].skew()
    save_dataframe(described.reset_index().rename(columns={"index": "feature"}),
                   out / "numeric_describe.csv")
    summary["numeric_describe"] = described.round(3).to_dict(orient="index")

    category_rows = []
    for column in categorical:
        for value, count in X[column].value_counts(dropna=False).sort_index().items():
            category_rows.append({
                "feature": column,
                "code": value,
                "level": _label(column, value),
                "count": int(count),
                "pct": round(100 * count / len(X), 2),
            })
    category_table = pd.DataFrame(category_rows)
    save_dataframe(category_table, out / "categorical_levels.csv")
    summary["categorical_levels"] = category_rows

    # -- 4. numeric distributions, split by outcome -------------------------
    n_cols = 3
    n_rows = int(np.ceil(len(numeric) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, numeric):
        for label, colour in ((0, "tab:blue"), (1, "tab:red")):
            values = X.loc[y == label, column].dropna()
            ax.hist(values, bins=20, alpha=0.55, color=colour, label=f"{target_name}={label}")
        ax.set_title(FEATURE_DESCRIPTIONS.get(column, column), fontsize=9)
        ax.set_xlabel(column)
        ax.legend(fontsize=7)
    for ax in axes[len(numeric):]:
        ax.set_visible(False)
    fig.suptitle("Numeric feature distributions by outcome", y=1.0)
    save_figure(fig, out / "numeric_distributions.png")

    # Boxplots make the location shift easier to read than overlaid histograms.
    fig, axes = plt.subplots(1, len(numeric), figsize=(2.6 * len(numeric), 3.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, numeric):
        ax.boxplot([X.loc[y == 0, column].dropna(), X.loc[y == 1, column].dropna()],
                   tick_labels=["no disease", "disease"], widths=0.6)
        ax.set_title(column, fontsize=10)
    fig.suptitle("Numeric features by outcome")
    save_figure(fig, out / "numeric_boxplots.png")

    # -- 5. correlation, numeric features ONLY ------------------------------
    corr = X[numeric].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(1.1 * len(numeric) + 2.5, 1.0 * len(numeric) + 2))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), numeric, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric)), numeric)
    for i in range(len(numeric)):
        for j in range(len(numeric)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation — NUMERIC features only\n"
                 "(categoricals excluded: their integer codes are not ordinal)",
                 fontsize=10)
    ax.grid(False)
    save_figure(fig, out / "correlation_numeric.png")
    summary["numeric_correlation"] = corr.round(3).to_dict()

    high_corr = [
        {"a": numeric[i], "b": numeric[j], "r": round(float(corr.iloc[i, j]), 3)}
        for i in range(len(numeric)) for j in range(i + 1, len(numeric))
        if abs(corr.iloc[i, j]) >= 0.7
    ]
    summary["highly_correlated_pairs"] = high_corr
    if high_corr:
        logger.warning("Highly correlated numeric pairs (|r| >= 0.7): %s", high_corr)

    # -- 6. association with the target, per feature type -------------------
    association_rows = []
    for column in numeric:
        mask = X[column].notna()
        r, p = stats.pointbiserialr(y[mask], X.loc[mask, column])
        association_rows.append({
            "feature": column, "type": "numeric", "statistic": "point-biserial r",
            "value": round(float(r), 3), "p_value": float(p),
        })
    for column in categorical:
        v = cramers_v(X[column].fillna(-1), y)
        table = pd.crosstab(X[column].fillna(-1), y)
        p = stats.chi2_contingency(table, correction=False)[1] if table.shape[0] > 1 else np.nan
        association_rows.append({
            "feature": column, "type": "categorical", "statistic": "Cramer's V",
            "value": round(float(v), 3), "p_value": float(p),
        })

    association = pd.DataFrame(association_rows)
    association["abs_value"] = association["value"].abs()
    association = association.sort_values("abs_value", ascending=False).drop(columns="abs_value")
    save_dataframe(association, out / "target_association.csv")
    summary["target_association"] = association.to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(association) + 1.6))
    colours = ["tab:blue" if t == "numeric" else "tab:orange"
               for t in association["type"]]
    ax.barh(association["feature"][::-1], association["value"].abs()[::-1],
            color=colours[::-1])
    ax.set_xlabel("|association with target|")
    ax.set_title("Univariate association with the target\n"
                 "blue = point-biserial r (numeric), orange = Cramer's V (categorical)",
                 fontsize=10)
    save_figure(fig, out / "target_association.png")

    # -- 7. categorical breakdown by outcome --------------------------------
    n_cols = 3
    n_rows = int(np.ceil(len(categorical) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, categorical):
        table = pd.crosstab(X[column], y, normalize="index")
        levels = [_label(column, v) for v in table.index]
        bottom = np.zeros(len(table))
        for label, colour in ((0, "tab:blue"), (1, "tab:red")):
            if label in table.columns:
                values = table[label].to_numpy()
                ax.bar(levels, values, bottom=bottom, color=colour,
                       label=f"{target_name}={label}")
                bottom += values
        ax.set_title(column, fontsize=10)
        ax.set_ylabel("proportion")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
    for ax in axes[len(categorical):]:
        ax.set_visible(False)
    fig.suptitle("Outcome rate within each categorical level", y=1.0)
    save_figure(fig, out / "categorical_vs_target.png")

    # -- 8. missingness -----------------------------------------------------
    missing = X.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(missing) + 1.8))
        ax.barh(missing.index, missing.to_numpy(), color="tab:grey")
        for i, value in enumerate(missing.to_numpy()):
            ax.text(value, i, f" {value} ({100 * value / len(X):.1f}%)", va="center", fontsize=9)
        ax.set_xlabel("missing values")
        ax.set_title("Missing values per feature")
        save_figure(fig, out / "missingness.png")
    summary["missing_per_feature"] = {k: int(v) for k, v in missing.items()}

    save_json(summary, out / "eda_summary.json")
    logger.info("EDA complete: %d figures/tables written to %s",
                len(list(out.glob('*'))), out)
    return summary
