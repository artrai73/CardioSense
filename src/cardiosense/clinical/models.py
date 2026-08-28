"""Model construction and hyperparameter search for the clinical pipeline.

Two models only, and the choice is deliberate:

* **Logistic Regression** — the interpretable baseline. Coefficients are log-odds
  ratios, which is the language clinical risk scores are already written in.
* **XGBoost** — the advanced model. It can represent the interactions a linear
  model cannot (e.g. ST depression mattering only in the presence of exercise
  angina), and it handles the mixed feature types natively.

Nothing else is added. Throwing five algorithms at 303 rows and reporting the
best test score is a multiple-comparisons problem dressed up as a benchmark: with
a ~45-patient test set, the winner is frequently decided by noise.

CatBoost was considered and rejected. Its advantage is native handling of
high-cardinality categorical features, and here the categoricals have 2-4 levels
each and are already one-hot encoded. It would add a heavy dependency for no
mechanism that applies to this data.

Search budget is kept small on purpose. ``RandomizedSearchCV`` with 40 candidates
and 5-fold stratified CV is 200 fits, roughly 30-60 seconds on Colab CPU. An
exhaustive grid over the same space would be ~10,000 fits, and on 212 training
rows the extra search would mostly be fitting the cross-validation noise.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from ..common.compat import logistic_penalty_kwargs
from ..common.config import Config
from ..common.logging_utils import get_logger

__all__ = ["build_logistic_regression", "tune_logistic_regression",
           "build_xgboost", "tune_xgboost", "SearchResult"]

logger = get_logger(__name__)


class SearchResult(dict):
    """Result of a hyperparameter search: best estimator, params, CV score, timing."""

    @property
    def estimator(self) -> Any:
        return self["estimator"]


def _cv(cfg: Config, n_splits: int) -> StratifiedKFold:
    """Stratified K-fold with a fixed seed, used for every search in this module."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.seed))


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
def build_logistic_regression(cfg: Config, **overrides: Any) -> LogisticRegression:
    """Construct an unfitted Logistic Regression from config.

    ``class_weight: balanced`` is set in the config. On a 46%/54% target this
    barely changes anything, but it makes the model's behaviour stable if you
    later switch to the pooled dataset, where the balance differs by site.
    """
    params = cfg.models.logistic_regression
    kwargs: dict[str, Any] = {
        "max_iter": int(params.get("max_iter", 2000)),
        "solver": params.get("solver", "lbfgs"),
        "class_weight": params.get("class_weight", None),
        "random_state": int(cfg.seed),
    }
    # sklearn 1.8 deprecated the `penalty` argument; the shim emits whatever the
    # installed version accepts, so this works on 1.4 through 1.10.
    kwargs.update(logistic_penalty_kwargs(params.get("penalty", "l2")))
    kwargs.update(overrides)
    return LogisticRegression(**kwargs)


def tune_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> SearchResult:
    """Tune the regularisation strength C with a small grid search.

    Only one hyperparameter is searched. C controls the L2 penalty, which is the
    single knob that matters on a small, mostly-linear problem; searching solvers
    or penalties as well would add fits without adding capability.

    Args:
        X_train: Preprocessed training matrix.
        y_train: Training labels.
        cfg: Clinical configuration.

    Returns:
        A :class:`SearchResult`.
    """
    params = cfg.models.logistic_regression
    grid = {"C": list(params.get("C_grid", [0.01, 0.1, 1.0, 10.0]))}
    folds = int(cfg.models.xgboost.get("cv_folds", 5))
    scoring = str(cfg.models.xgboost.get("scoring", "roc_auc"))

    logger.info("Tuning Logistic Regression: %d candidates x %d folds = %d fits",
                len(grid["C"]), folds, len(grid["C"]) * folds)

    search = GridSearchCV(
        estimator=build_logistic_regression(cfg),
        param_grid=grid,
        scoring=scoring,
        cv=_cv(cfg, folds),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info("Logistic Regression best C=%s, CV %s=%.4f (+/- %.4f), %.1fs",
                search.best_params_["C"], scoring, search.best_score_,
                search.cv_results_["std_test_score"][search.best_index_], elapsed)

    return SearchResult({
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_score": float(search.best_score_),
        "cv_score_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "cv_train_score": float(search.cv_results_["mean_train_score"][search.best_index_]),
        "scoring": scoring,
        "n_fits": int(len(grid["C"]) * folds),
        "search_seconds": round(elapsed, 2),
        "search_type": "GridSearchCV",
    })


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def build_xgboost(cfg: Config, **overrides: Any) -> Any:
    """Construct an unfitted ``XGBClassifier`` from config.

    Raises:
        ImportError: If xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required. Run: pip install xgboost") from exc

    fixed = dict(cfg.models.xgboost.get("fixed_params", {}))
    kwargs: dict[str, Any] = {
        "objective": fixed.get("objective", "binary:logistic"),
        "eval_metric": fixed.get("eval_metric", "logloss"),
        "tree_method": fixed.get("tree_method", "hist"),
        "n_jobs": int(fixed.get("n_jobs", -1)),
        "random_state": int(cfg.seed),
    }
    kwargs.update(overrides)
    return XGBClassifier(**kwargs)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> SearchResult:
    """Tune XGBoost with a budgeted randomized search.

    The search space in the config is constrained to shallow trees
    (``max_depth`` 2-4) with strong subsampling. With 212 training rows, depth-8
    trees memorise the training set within a handful of boosting rounds; the
    cross-validation score then measures how well the model memorised, not how
    well it generalises.

    Args:
        X_train: Preprocessed training matrix.
        y_train: Training labels.
        cfg: Clinical configuration.

    Returns:
        A :class:`SearchResult`.
    """
    params = cfg.models.xgboost
    space = {key: list(values) for key, values in params.search_space.items()}
    n_iter = int(params.get("n_iter_search", 40))
    folds = int(params.get("cv_folds", 5))
    scoring = str(params.get("scoring", "roc_auc"))

    total_combinations = int(np.prod([len(v) for v in space.values()]))
    logger.info(
        "Tuning XGBoost: sampling %d of %d possible combinations x %d folds = %d fits "
        "(exhaustive grid would be %d fits)",
        n_iter, total_combinations, folds, n_iter * folds, total_combinations * folds,
    )

    search = RandomizedSearchCV(
        estimator=build_xgboost(cfg),
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=_cv(cfg, folds),
        n_jobs=-1,
        random_state=int(cfg.seed),
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_index = search.best_index_
    cv_std = float(search.cv_results_["std_test_score"][best_index])
    cv_train = float(search.cv_results_["mean_train_score"][best_index])

    logger.info("XGBoost best CV %s=%.4f (+/- %.4f), train %s=%.4f, %.1fs",
                scoring, search.best_score_, cv_std, scoring, cv_train, elapsed)
    logger.info("Best params: %s", search.best_params_)

    if cv_train - search.best_score_ > 0.15:
        logger.warning(
            "Train-CV gap is %.3f — the model is overfitting even after tuning. "
            "Expect the simpler baseline to be competitive.",
            cv_train - search.best_score_,
        )

    return SearchResult({
        "estimator": search.best_estimator_,
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "cv_score": float(search.best_score_),
        "cv_score_std": cv_std,
        "cv_train_score": cv_train,
        "scoring": scoring,
        "n_fits": int(n_iter * folds),
        "search_space_size": total_combinations,
        "search_seconds": round(elapsed, 2),
        "search_type": "RandomizedSearchCV",
    })
