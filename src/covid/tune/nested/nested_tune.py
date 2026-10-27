from functools import partial
from typing import Any

import pandas as pd
from loguru import logger
from optuna_integration import OptunaSearchCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from covid.common import constants
from covid.common.data import load_and_split_data
from covid.tune import TuningSpec
from covid.tune.nested.result import NestedTuningResult
from covid.tune.nested.spec import NestedTuningSpec
from covid.tune.nested.tracking import NestedTuningTracker


def nested_tune(
    spec: NestedTuningSpec, tracker: NestedTuningTracker
) -> NestedTuningResult:
    tracker.track_spec(spec)
    result = _nested_tune(spec)
    tracker.track_result(result)
    return result


def _nested_tune(spec: NestedTuningSpec) -> NestedTuningResult:
    X, y = load_and_split_data(spec.inner.data_path)

    inner_search = _create_inner_search(spec.inner)
    outer_cv = _create_outer_cv(spec.n_outer_splits, spec.n_outer_repeats)
    nested_scores = _run_nested_cv(X, y, inner_search, outer_cv, spec.inner.scoring)

    return NestedTuningResult.from_nested_scores(nested_scores)


def _create_inner_search(spec: TuningSpec) -> OptunaSearchCV:
    cv = StratifiedKFold(
        n_splits=spec.n_splits, shuffle=True, random_state=constants.RANDOM_STATE
    )
    return OptunaSearchCV(
        estimator=spec.pipeline,
        param_distributions=spec.param_distributions,
        n_trials=spec.n_searches,
        scoring=spec.scoring[0],
        refit=True,
        cv=cv,
        return_train_score=True,
        random_state=constants.RANDOM_STATE,
        n_jobs=1,
        enable_pruning=False,
    )


def _create_outer_cv(n_splits: int, n_repeats: int = 1) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=constants.RANDOM_STATE
    )


def _run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    inner_search: OptunaSearchCV,
    outer_cv: RepeatedStratifiedKFold,
    scoring: list[str],
) -> dict[str, Any]:
    logger.info(
        "Running nested cross-validation with {} outer splits and {} inner splits",
        outer_cv.get_n_splits(),
        inner_search.cv.get_n_splits(),
    )
    return cross_validate(
        inner_search,
        X,
        y,
        cv=outer_cv,
        scoring=partial(_score_best_estimator, scoring=scoring),
        return_estimator=True,
        n_jobs=-1,
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
