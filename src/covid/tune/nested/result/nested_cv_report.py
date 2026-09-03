from typing import Any

import pandas as pd

from covid.tune.common.cv_report.fitted_search_cv import FittedSearchCV


class NestedCVReport:
    def __init__(self, cross_validated_scores: dict[str, Any]) -> None:
        self._validate_cv_scores(cross_validated_scores)
        self._cv_scores = pd.DataFrame(cross_validated_scores)

    @staticmethod
    def _validate_cv_scores(cross_validated_scores: dict[str, Any]) -> None:
        if "estimator" not in cross_validated_scores:
            raise ValueError(
                "NestedCVReport requires cross_validate(..., return_estimator=True)"
            )
        if not all(
                isinstance(est, FittedSearchCV)
                for est in cross_validated_scores["estimator"]
        ):
            raise ValueError(
                "All estimators must be instances of SearchCV — "
                "did you forget to pass a Search object as the inner CV in cross_validate()?"
            )

    def to_dataframe(self) -> pd.DataFrame:
        df = self._cv_scores.drop(columns=["estimator"])
        df["params"] = self.params_per_fold()
        return df

    def mean_test_scores(self) -> dict[str, float]:
        return self._cv_scores.filter(like="test_").mean().to_dict()

    def mean_train_scores(self) -> dict[str, float]:
        return self._cv_scores.filter(like="train_").mean().to_dict()

    def std_test_scores(self) -> dict[str, float]:
        return self._cv_scores.filter(like="test_").std().to_dict()

    def std_train_scores(self) -> dict[str, float]:
        return self._cv_scores.filter(like="train_").std().to_dict()

    def params_per_fold(self) -> list[dict[str, Any]]:
        return self._cv_scores["estimator"].apply(lambda est: est.best_params_).tolist()
