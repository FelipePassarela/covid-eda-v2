from unittest.mock import MagicMock

import pandas as pd
import pytest

from covid.eval.context import EvaluationContext
from covid.eval.evaluate import _evaluate_pipeline


def test_evaluation_preserves_indices(
    evaluation_context: EvaluationContext, prediction_pipeline: MagicMock
) -> None:
    result = _evaluate_pipeline(evaluation_context)
    expected_predictions = pd.Series([0, 1, 0, 1], index=evaluation_context.y.index)

    pd.testing.assert_series_equal(result.y_true, evaluation_context.y)
    pd.testing.assert_series_equal(result.y_pred, expected_predictions)
    assert result.y_score is not None
    pd.testing.assert_index_equal(result.y_score.index, evaluation_context.y.index)
    prediction_pipeline.predict.assert_called_once_with(evaluation_context.X)


def test_evaluation_uses_positive_class_probabilities(
    evaluation_context: EvaluationContext, prediction_pipeline: MagicMock
) -> None:
    result = _evaluate_pipeline(evaluation_context)
    expected_scores = pd.Series([0.1, 0.4, 0.35, 0.8], index=evaluation_context.y.index)

    pd.testing.assert_series_equal(result.y_score, expected_scores)
    assert result.roc_auc == pytest.approx(0.75)
    prediction_pipeline.predict_proba.assert_called_once_with(evaluation_context.X)
