import pandas as pd

from covid.tune.common.cv_report import FittedSearchCV


class SearchCVReport:
    def __init__(self, search: FittedSearchCV, include_std: bool = False) -> None:
        self._cv_results = pd.DataFrame(search.cv_results_)
        self._include_std = include_std

    def to_dataframe(self) -> pd.DataFrame:
        columns_of_interest = self._columns_of_interest()
        return self._cv_results[columns_of_interest]

    def _columns_of_interest(self) -> list[str]:
        metrics_columns = []

        for metric in self._cv_results_metrics():
            candidates = [
                f"mean_test_{metric}",
                f"std_test_{metric}" if self._include_std else None,
                f"mean_train_{metric}",
                f"std_train_{metric}" if self._include_std else None,
            ]
            candidates = [col for col in candidates if col is not None]
            metrics_columns.extend(
                str(col) for col in candidates if col in self._cv_results.columns
            )

        return ["params"] + metrics_columns

    def _cv_results_metrics(self) -> list[str]:
        return [
            column.removeprefix("mean_test_")
            for column in self._cv_results.columns
            if column.startswith("mean_test_")
        ]
