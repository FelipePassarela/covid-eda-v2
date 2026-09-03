from sklearn.callback import ProgressBar
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
)

from covid.common import constants
from covid.common.data import load_and_split_data
from covid.tune.base import TuningResult
from covid.tune.base.spec import TuningSpec
from covid.tune.base.tracking import TuningTracker


def tune(spec: TuningSpec, tracker: TuningTracker) -> TuningResult:
    tracker.track_spec(spec)
    result = _tune(spec)
    tracker.track_result(result)
    return result


def _tune(spec: TuningSpec) -> TuningResult:
    X, y = load_and_split_data(spec.data_path)
    search = _create_search(spec)
    search.fit(X, y)
    return TuningResult.from_fitted_search(search)


def _create_search(spec: TuningSpec) -> RandomizedSearchCV:
    cv = RepeatedStratifiedKFold(
        n_splits=spec.n_splits,
        n_repeats=spec.n_fold_repeats,
        random_state=constants.RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        estimator=spec.pipeline,
        param_distributions=spec.param_distributions,
        n_iter=spec.n_searches,
        scoring=spec.scoring,
        refit=spec.scoring[0],
        cv=cv,
        return_train_score=True,
        verbose=1,
        n_jobs=-1,
        random_state=constants.RANDOM_STATE,
    )
    search.set_callbacks(ProgressBar())
    return search
