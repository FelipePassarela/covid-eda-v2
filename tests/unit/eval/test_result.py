import math

import numpy as np
import pandas as pd
import pytest

from covid_model.eval.result import EvaluationResult


def test_from_predictions_computes_metrics() -> None:
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 1, 1, 0, 1])
    y_score = pd.Series([0.1, 0.4, 0.6, 0.7, 0.3, 0.8])

    result = EvaluationResult.from_predictions(y_true, y_pred, y_score)

    assert result.balanced_accuracy == pytest.approx(0.375)
    assert result.recall == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.25)
    assert result.f1 == pytest.approx(1 / 3)
    assert result.roc_auc == pytest.approx(0.625)

    np.testing.assert_array_equal(result.confusion_matrix, [[1, 3], [1, 1]])


def test_from_predictions_preserves_series() -> None:
    index = pd.Index([42, 7, 99, 12], name="sample")
    y_true = pd.Series([0, 0, 1, 1], index=index, name="actual")
    y_pred = pd.Series([0, 1, 0, 1], index=index, name="predicted")
    y_score = pd.Series([0.1, 0.4, 0.35, 0.8], index=index, name="probability")

    result = EvaluationResult.from_predictions(y_true, y_pred, y_score)

    pd.testing.assert_series_equal(result.y_true, y_true)
    pd.testing.assert_series_equal(result.y_pred, y_pred)
    pd.testing.assert_series_equal(result.y_score, y_score)


def test_from_predictions_without_scores_marks_roc_auc_as_nan() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 0, 1])

    result = EvaluationResult.from_predictions(y_true, y_pred)

    assert result.y_score is None
    assert math.isnan(result.roc_auc)
    assert result.balanced_accuracy == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    np.testing.assert_array_equal(result.confusion_matrix, [[1, 1], [1, 1]])
