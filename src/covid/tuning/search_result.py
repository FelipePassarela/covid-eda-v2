from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class HyperparameterSearchResult:
    best_estimator: BaseEstimator
    best_params: dict[str, Any]
    best_score: float
    report: pd.DataFrame

    @staticmethod
    def report_from_cv_results(
        cv_results: dict[str, Any],
        sort_by: str = "balanced_accuracy",
        include_std: bool = False,
    ) -> pd.DataFrame:
        results = pd.DataFrame(cv_results)

        metrics = [
            column.removeprefix("mean_test_")
            for column in results.columns
            if column.startswith("mean_test_")
        ]
        cv_cols = ["params"]

        for metric in metrics:
            candidates = [
                f"mean_test_{metric}",
                f"std_test_{metric}" if include_std else None,
                f"mean_train_{metric}",
                f"std_train_{metric}" if include_std else None,
            ]
            cv_cols.extend(
                str(column)
                for column in candidates
                if column is not None and column in results.columns
            )

        sort_column = f"mean_test_{sort_by}"

        return (
            results[cv_cols]
            .sort_values(sort_column, ascending=False)
            .reset_index(drop=True)
        )
