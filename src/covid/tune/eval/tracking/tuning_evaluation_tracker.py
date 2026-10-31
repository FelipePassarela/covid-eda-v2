from typing import Protocol

from covid.tune.eval.result import TuningEvaluationResult
from covid.tune.eval.tuning_evaluation_spec import TuningEvaluationSpec


class TuningEvaluationTracker(Protocol):
    """Record a tuning evaluation's configuration and performance estimates."""

    def track_spec(self, spec: TuningEvaluationSpec) -> None:
        pass

    def track_result(self, result: TuningEvaluationResult) -> None:
        pass
