from typing import Protocol

from covid_tuning.search.result import TuningResult
from covid_tuning.search.tuning_spec import TuningSpec


class TuningTracker(Protocol):
    def track_spec(self, spec: TuningSpec) -> None:
        pass

    def track_result(self, result: TuningResult) -> None:
        pass
