import pandas as pd
from sklearn.callback import ProgressBar
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from covid.common import constants
from covid.tune.search_result import HyperparameterSearchResult
from covid.tune.search_spec import RandomizedSearchSpec


def search_hyperparameters(
    X: pd.DataFrame, y: pd.Series, spec: RandomizedSearchSpec
) -> HyperparameterSearchResult:
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

    search.fit(X, y)

    return HyperparameterSearchResult(
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_score=search.best_score_,
        report=HyperparameterSearchResult.report_from_cv_results(search.cv_results_),
    )
