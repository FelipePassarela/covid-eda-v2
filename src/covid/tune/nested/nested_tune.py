from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from covid.common import constants
from covid.common.data import load_and_split_data
from covid.tune import TuningSpec
from covid.tune.nested.nested_tuning_spec import NestedTuningSpec
from covid.tune.nested.result import NestedTuningResult
from covid.tune.nested.tracking import NestedTuningTracker


def nested_tune(spec: NestedTuningSpec, tracker: NestedTuningTracker) -> None:
    tracker.track_spec(spec)
    result = _nested_tune(spec)
    tracker.track_result(result)


def _nested_tune(spec: NestedTuningSpec) -> NestedTuningResult:
    X, y = load_and_split_data(spec.inner.data_path)

    inner_search = _create_inner_search(spec.inner)
    outer_cv = _create_outer_cv(spec.outer_n_splits)
    nested_scores = _run_nested_cv(X, y, inner_search, outer_cv)

    return NestedTuningResult.from_nested_scores(nested_scores)


def _create_inner_search(spec: TuningSpec) -> RandomizedSearchCV:
    cv = StratifiedKFold(
        n_splits=spec.n_splits, shuffle=True, random_state=constants.RANDOM_STATE
    )
    return RandomizedSearchCV(
        estimator=spec.pipeline,
        param_distributions=spec.param_distributions,
        n_iter=spec.n_searches,
        scoring=spec.scoring,
        refit=spec.scoring[0],
        cv=cv,
        return_train_score=True,
        random_state=constants.RANDOM_STATE,
    )


def _create_outer_cv(outer_n_splits: int) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=outer_n_splits, shuffle=True, random_state=constants.RANDOM_STATE
    )


def _run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    inner_search: RandomizedSearchCV,
    outer_cv: StratifiedKFold,
) -> dict[str, Any]:
    return cross_validate(
        inner_search,
        X,
        y,
        cv=outer_cv,
        scoring=inner_search.scoring,
        return_estimator=True,
        n_jobs=-1,
        verbose=1,
    )
