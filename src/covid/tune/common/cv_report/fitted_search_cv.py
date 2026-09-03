from typing import Any, Protocol, runtime_checkable

from sklearn.base import BaseEstimator


@runtime_checkable
class FittedSearchCV(Protocol):
    @property
    def best_params_(self) -> dict[str, Any]:
        pass

    @property
    def best_estimator_(self) -> BaseEstimator:
        pass

    @property
    def best_score_(self) -> float:
        pass

    @property
    def cv_results_(self) -> dict[str, Any]:
        pass
