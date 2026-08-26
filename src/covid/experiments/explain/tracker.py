from typing import Protocol

from covid.experiments.explain.result import ExplainingResult
from covid.experiments.explain.spec import ExplainingSpec


class ExplainingTracker(Protocol):
    def track_spec(self, spec: ExplainingSpec) -> None:
        pass

    def track_result(self, result: ExplainingResult) -> None:
        pass
