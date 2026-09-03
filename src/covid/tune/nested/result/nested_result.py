from dataclasses import dataclass
from typing import Any, Self

import pandas as pd

from covid.tune.common.cv_report import NestedCVReport


@dataclass(frozen=True)
class NestedTuningResult:
    mean_test_scores: dict[str, float]
    mean_train_scores: dict[str, float]
    std_test_scores: dict[str, float]
    std_train_scores: dict[str, float]
    params_per_fold: list[dict[str, Any]]
    cv_report: pd.DataFrame

    @classmethod
    def from_nested_scores(cls, nested_scores: dict[str, Any]) -> Self:
        nested_cv_report = NestedCVReport(nested_scores)
        return cls(
            mean_test_scores=nested_cv_report.mean_test_scores(),
            mean_train_scores=nested_cv_report.mean_train_scores(),
            std_test_scores=nested_cv_report.std_test_scores(),
            std_train_scores=nested_cv_report.std_train_scores(),
            params_per_fold=nested_cv_report.params_per_fold(),
            cv_report=nested_cv_report.to_dataframe(),
        )
