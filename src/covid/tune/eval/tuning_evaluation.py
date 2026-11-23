"""Estimate tuning performance on unseen data via nested cross-validation."""

from functools import partial
from typing import Any

import pandas as pd
from loguru import logger
from optuna_integration import OptunaSearchCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

from covid.common.data import load_and_split_data
from covid.tune.eval.result import TuningEvaluationResult
from covid.tune.eval.tracking import TuningEvaluationTracker
from covid.tune.eval.tuning_evaluation_spec import TuningEvaluationSpec
from covid.tune.search.tuning import build_optuna_search


def evaluate_tuning(
    spec: TuningEvaluationSpec, tracker: TuningEvaluationTracker
) -> TuningEvaluationResult:
    """Evaluate and track a tuning procedure across held-out outer folds.

    Each outer fold runs its own inner hyperparameter search. The result
    contains aggregate scores, per-fold parameters, and a fold report.
    """
    tracker.track_spec(spec)
    result = _evaluate_tuning(spec)
    tracker.track_result(result)
    return result


def _evaluate_tuning(spec: TuningEvaluationSpec) -> TuningEvaluationResult:
    """Estimate the configured tuning procedure's generalization performance."""
    X, y = load_and_split_data(spec.inner.data_path)
    nested_scores = _run_nested_cv(X, y, spec)
    return TuningEvaluationResult.from_nested_scores(nested_scores)


def _run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    spec: TuningEvaluationSpec,
) -> dict[str, Any]:
    inner_search = build_optuna_search(spec.inner)
    scorer = partial(_score_best_estimator, scoring=spec.inner.scoring)
    logger.info(
        "Evaluating tuning via nested cross-validation with {} outer evaluation "
        "splits and {} inner tuning splits",
        spec.outer_cv.get_n_splits(),
        spec.inner.cv.get_n_splits(),
    )
    return cross_validate(
        inner_search,
        X,
        y,
        cv=spec.outer_cv,
        scoring=scorer,
        return_estimator=True,
        n_jobs=spec.n_outer_jobs,
        verbose=3,
        error_score="raise",
    )


def _score_best_estimator(
    search: OptunaSearchCV, X_test: pd.DataFrame, y_test: pd.Series, scoring: list[str]
) -> dict[str, float]:
    # OptunaSearchCV exposes decision_function even when its pipeline does
    # not. Score the fitted pipeline so sklearn selects a supported method.
    scorers = {metric: get_scorer(metric) for metric in scoring}
    return {
        metric: scorer(search.best_estimator_, X_test, y_test)
        for metric, scorer in scorers.items()
    }
