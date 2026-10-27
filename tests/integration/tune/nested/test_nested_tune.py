from unittest.mock import MagicMock

import numpy as np
import pytest
from imblearn.pipeline import Pipeline
from optuna.distributions import CategoricalDistribution
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from covid.tune import TuningSpec
from covid.tune.nested import NestedTuningSpec
from covid.tune.nested.nested_tune import nested_tune
from tests.integration.tune.nested.conftest import assert_nested_result


def test_nested_tune_succeeds_with_valid_spec(nested_spec: NestedTuningSpec) -> None:
    result = nested_tune(nested_spec, tracker=MagicMock())
    assert_nested_result(result, nested_spec)


@pytest.mark.parametrize(
    "classifier,parameter,value",
    [
        (DummyClassifier(), "strategy", "prior"),
        (XGBClassifier(n_estimators=5, n_jobs=1, random_state=42), "max_depth", 2),
        (LinearSVC(random_state=42), "C", 1.0),
    ],
    ids=["dummy-probabilities", "xgboost-probabilities", "svm-decision-function"],
)
def test_nested_tune_computes_metrics_with_proba_or_decision_function(
    nested_spec: NestedTuningSpec, classifier, parameter: str, value
) -> None:
    metrics = ["roc_auc", "average_precision", "balanced_accuracy"]
    if hasattr(classifier, "predict_proba"):
        metrics.append("neg_log_loss")
    nested_spec = NestedTuningSpec(
        inner=TuningSpec(
            pipeline=Pipeline([("clf", classifier)]),
            param_distributions={f"clf__{parameter}": CategoricalDistribution([value])},
            data_path=nested_spec.inner.data_path,
            n_searches=1,
            n_splits=2,
            n_fold_repeats=1,
            scoring=metrics,
        ),
        n_outer_splits=nested_spec.n_outer_splits,
        n_outer_repeats=nested_spec.n_outer_repeats,
    )

    result = nested_tune(nested_spec, tracker=MagicMock())

    assert_nested_result(result, nested_spec)
    scores = result.cv_report.filter(like="test_")
    assert np.isfinite(scores.to_numpy()).all()
    assert np.isfinite(list(result.mean_test_scores.values())).all()
    assert scores["test_roc_auc"].between(0, 1).all()
    assert scores["test_average_precision"].between(0, 1).all()
