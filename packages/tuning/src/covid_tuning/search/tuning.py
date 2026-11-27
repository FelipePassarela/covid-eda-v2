from optuna_integration import OptunaSearchCV

from covid_common import constants
from covid_common.data import load_and_split_data
from covid_tuning.common.cv_report import OptunaSearchCVAdapter
from covid_tuning.search.result import TuningResult
from covid_tuning.search.tracking import TuningTracker
from covid_tuning.search.tuning_spec import TuningSpec


def tune(spec: TuningSpec, tracker: TuningTracker) -> TuningResult:
    tracker.track_spec(spec)
    result = _tune(spec)
    tracker.track_result(result)
    return result


def _tune(spec: TuningSpec) -> TuningResult:
    X, y = load_and_split_data(spec.data_path)
    search = build_optuna_search(spec)
    search.fit(X, y)
    fitted_search = OptunaSearchCVAdapter(search, spec.scoring[0])
    return TuningResult.from_fitted_search(fitted_search)


def build_optuna_search(spec: TuningSpec) -> OptunaSearchCV:
    # noinspection bad-argument-type
    return OptunaSearchCV(
        estimator=spec.pipeline,
        param_distributions=spec.param_distributions,
        n_trials=spec.n_searches,
        scoring=spec.scoring[0],
        cv=spec.cv,
        refit=True,
        return_train_score=True,
        enable_pruning=False,
        random_state=constants.RANDOM_STATE,
        n_jobs=spec.n_jobs,
    )
