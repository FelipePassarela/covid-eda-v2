from typing import Protocol

from covid.tune.nested.nested_tuning_spec import NestedTuningSpec
from covid.tune.nested.result.nested_result import NestedTuningResult


class NestedTuningTracker(Protocol):
    def track_spec(self, spec: NestedTuningSpec) -> None:
        pass

    def track_result(self, result: NestedTuningResult) -> None:
        pass
