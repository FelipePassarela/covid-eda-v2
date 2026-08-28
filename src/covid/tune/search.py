from sklearn.callback import ProgressBar
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from covid.common import constants
from covid.common.data import load_and_split_data
from covid.tune.search_result import HyperparameterSearchResult
from covid.tune.search_spec import RandomizedSearchSpec


def search_hyperparameters(spec: RandomizedSearchSpec) -> HyperparameterSearchResult:
    X, y = load_and_split_data(spec.data_path)
    search = _create_search(spec)
    search.fit(X, y)
    return HyperparameterSearchResult.from_fitted_search(search)


def _create_search(spec: RandomizedSearchSpec) -> RandomizedSearchCV:
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=spec.n_fold_repeats, random_state=constants.RANDOM_STATE
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
