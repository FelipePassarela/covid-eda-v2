from typing import Any

import pandas as pd


class CVReport:
    def __init__(
        self,
        cv_results: dict[str, Any],
        include_std: bool = False,
        sort_by: str | None = None,
    ) -> None:
        self._cv_results = pd.DataFrame(cv_results)
        self._include_std = include_std
        self._sort_by = sort_by

    def to_dataframe(self) -> pd.DataFrame:
        columns_of_interest = self._columns_of_interest()
        column_to_sort_by = self._column_to_sort_by(columns_of_interest)
        return self._create_report(columns_of_interest, column_to_sort_by)

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

    def _column_to_sort_by(self, cols_of_interest: list[str]) -> str:
        return (
            f"mean_test_{self._sort_by}"
            if self._sort_by is not None
            else cols_of_interest[1]
        )

    def _create_report(
        self, cols_of_interest: list[str], sort_by_column: str
    ) -> pd.DataFrame:
        return (
            self._cv_results[cols_of_interest]
            .sort_values(sort_by_column, ascending=False)
            .reset_index(drop=True)
        )
