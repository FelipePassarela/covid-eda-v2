from unittest.mock import MagicMock

import numpy as np

from covid.tune.nested import NestedTuningSpec
from covid.tune.nested.nested_tune import nested_tune
from tests.integration.tune.nested.conftest import assert_nested_result


def test_nested_tune_succeeds_with_valid_spec(nested_spec: NestedTuningSpec) -> None:
    result = nested_tune(nested_spec, tracker=MagicMock())
    assert_nested_result(result, nested_spec)


def test_nested_tune_computes_metrics_with_proba_or_decision_function(
    nested_scoring_spec: NestedTuningSpec,
) -> None:
    result = nested_tune(nested_scoring_spec, tracker=MagicMock())

    assert_nested_result(result, nested_scoring_spec)
    scores = result.cv_report.filter(like="test_")
    assert np.isfinite(scores.to_numpy()).all()
    assert np.isfinite(list(result.mean_test_scores.values())).all()
    assert scores["test_roc_auc"].between(0, 1).all()
    assert scores["test_average_precision"].between(0, 1).all()
