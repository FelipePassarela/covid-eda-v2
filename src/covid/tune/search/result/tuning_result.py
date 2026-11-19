from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from sklearn.base import BaseEstimator

from covid.tune.common.cv_report import FittedSearchCV
from covid.tune.search.result.search_cv_report import SearchCVReport


@dataclass(frozen=True)
class TuningResult:
    best_estimator: BaseEstimator
    best_params: dict[str, Any]
    best_score: float
    cv_report: pd.DataFrame

    @classmethod
    def from_fitted_search(cls, search: FittedSearchCV) -> Self:
        cv_report = SearchCVReport(search, include_std=True).to_dataframe()
        return cls(
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_report=cv_report,
        )
