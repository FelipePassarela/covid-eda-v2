from typing import Protocol

from covid.tune.search.result import TuningResult
from covid.tune.search.tuning_spec import TuningSpec


class TuningTracker(Protocol):
    def track_spec(self, spec: TuningSpec) -> None:
        pass

    def track_result(self, result: TuningResult) -> None:
        pass
