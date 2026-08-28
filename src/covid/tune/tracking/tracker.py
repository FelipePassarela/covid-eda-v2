from typing import Protocol

from covid.tune import TuningResult, TuningSpec


class TuningTracker(Protocol):
    def track_spec(self, spec: TuningSpec) -> None:
        pass

    def track_result(self, result: TuningResult) -> None:
        pass
