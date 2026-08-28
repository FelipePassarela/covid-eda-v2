from typing import Protocol

from covid.tune import TuningResult, TuningSpec


class HyperparameterSearchTracker(Protocol):
    def track_search(self, spec: TuningSpec, result: TuningResult) -> None: ...
