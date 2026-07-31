from typing import Protocol

from covid.tuning import HyperparameterSearchResult, RandomizedSearchSpec


class HyperparameterSearchTracker(Protocol):
    def track_result(self, result: HyperparameterSearchResult) -> None: ...
    def track_spec(self, spec: RandomizedSearchSpec) -> None: ...
