from typing import Protocol

from covid.tune import HyperparameterSearchResult, RandomizedSearchSpec


class HyperparameterSearchTracker(Protocol):
    def track_search(
        self, spec: RandomizedSearchSpec, result: HyperparameterSearchResult
    ) -> None: ...
