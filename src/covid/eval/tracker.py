from pathlib import Path
from typing import Protocol

from covid.eval.context import EvaluationContext
from covid.eval.result import EvaluationResult


class EvaluationTracker(Protocol):
    def track_spec(self, pipeline_path: Path, data_path: Path) -> None:
        pass

    def track_context(self, context: EvaluationContext) -> None:
        pass

    def track_result(self, result: EvaluationResult) -> None:
        pass
