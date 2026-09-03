from typing import Protocol

from covid.tune.base.spec import TuningSpec
from covid.tune.base.tune import TuningResult


class TuningTracker(Protocol):
    def track_spec(self, spec: TuningSpec) -> None:
        pass

    def track_result(self, result: TuningResult) -> None:
        pass
