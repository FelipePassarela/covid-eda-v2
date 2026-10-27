import pytest
from imblearn.pipeline import Pipeline
from optuna.distributions import CategoricalDistribution
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from covid.tune import TuningSpec
from covid.tune.nested import NestedTuningResult, NestedTuningSpec


@pytest.fixture()
def nested_spec(spec: TuningSpec) -> NestedTuningSpec:
    return NestedTuningSpec(inner=spec, n_outer_splits=3, n_outer_repeats=2)


@pytest.fixture(
    params=[
        pytest.param(
            (DummyClassifier(), "strategy", "prior"), id="dummy-probabilities"
        ),
        pytest.param(
            (XGBClassifier(n_estimators=5, n_jobs=1, random_state=42), "max_depth", 2),
            id="xgboost-probabilities",
        ),
        pytest.param(
            (LinearSVC(random_state=42), "C", 1.0), id="svm-decision-function"
        ),
    ]
)
def nested_scoring_spec(
    request: pytest.FixtureRequest, nested_spec: NestedTuningSpec
) -> NestedTuningSpec:
    classifier, parameter, value = request.param
    metrics = ["roc_auc", "average_precision", "balanced_accuracy"]
    if hasattr(classifier, "predict_proba"):
        metrics.append("neg_log_loss")

    return NestedTuningSpec(
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


def assert_nested_result(result: NestedTuningResult, spec: NestedTuningSpec) -> None:
    assert result.mean_test_scores
    assert result.std_test_scores
    assert set(result.mean_test_scores) == {f"test_{s}" for s in spec.inner.scoring}
    assert set(result.std_test_scores) == {f"test_{s}" for s in spec.inner.scoring}

    assert len(result.params_per_fold) == spec.n_outer_splits * spec.n_outer_repeats
    assert not result.cv_report.empty
    assert len(result.cv_report) == len(result.params_per_fold)
