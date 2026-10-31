"""Evaluate tuning procedures with nested cross-validation.

Use covid.eval to evaluate a final fitted model.
"""

from .result import TuningEvaluationResult
from .tuning_evaluation import evaluate_tuning
from .tuning_evaluation_spec import TuningEvaluationSpec

__all__ = ["TuningEvaluationResult", "TuningEvaluationSpec", "evaluate_tuning"]
