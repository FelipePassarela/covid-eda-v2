from unittest.mock import MagicMock

import numpy as np

from covid_tuning.eval import TuningEvaluationResult, TuningEvaluationSpec
from covid_tuning.eval.tuning_evaluation import evaluate_tuning


def test_evaluate_tuning_succeeds_with_valid_spec(
    evaluation_spec: TuningEvaluationSpec,
) -> None:
    result = evaluate_tuning(evaluation_spec, tracker=MagicMock())
    _assert_evaluation_result(result, evaluation_spec)


def test_evaluate_tuning_computes_metrics_with_proba_or_decision_function(
    evaluation_scoring_spec: TuningEvaluationSpec,
) -> None:
    result = evaluate_tuning(evaluation_scoring_spec, tracker=MagicMock())

    _assert_evaluation_result(result, evaluation_scoring_spec)
    scores = result.cv_report.filter(like="test_")
    assert np.isfinite(scores.to_numpy()).all()
    assert np.isfinite(list(result.mean_test_scores.values())).all()
    assert scores["test_roc_auc"].between(0, 1).all()
    assert scores["test_average_precision"].between(0, 1).all()


def _assert_evaluation_result(
    result: TuningEvaluationResult, spec: TuningEvaluationSpec
) -> None:
    assert result.mean_test_scores
    assert result.std_test_scores
    assert set(result.mean_test_scores) == {f"test_{s}" for s in spec.inner.scoring}
    assert set(result.std_test_scores) == {f"test_{s}" for s in spec.inner.scoring}

    assert len(result.params_per_fold) == spec.outer_cv.get_n_splits()
    assert not result.cv_report.empty
    assert len(result.cv_report) == len(result.params_per_fold)
