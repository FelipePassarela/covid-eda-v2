from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from covid.tune.cv_report import CVReport


@dataclass(frozen=True)
class TuningResult:
    best_estimator: BaseEstimator
    best_params: dict[str, Any]
    best_score: float
    cv_report: pd.DataFrame

    @classmethod
    def from_fitted_search(cls, search: RandomizedSearchCV) -> Self:
        cv_report = CVReport(
            search.cv_results_, include_std=True, sort_by=search.scoring[0]
        )
        return cls(
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_report=cv_report.to_dataframe(),
        )
