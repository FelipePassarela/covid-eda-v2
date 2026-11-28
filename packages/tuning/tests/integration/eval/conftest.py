import pytest
from imblearn.pipeline import Pipeline
from optuna.distributions import CategoricalDistribution
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from covid_common.constants import RANDOM_STATE
from covid_tuning import TuningSpec
from covid_tuning.eval import TuningEvaluationSpec


@pytest.fixture()
def evaluation_spec(spec: TuningSpec) -> TuningEvaluationSpec:
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner = spec.model_copy(update={"cv": inner_cv})
    outer_cv = RepeatedStratifiedKFold(
        n_splits=3, n_repeats=2, random_state=RANDOM_STATE
    )
    return TuningEvaluationSpec(inner=inner, outer_cv=outer_cv)


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
def evaluation_scoring_spec(
    request: pytest.FixtureRequest, evaluation_spec: TuningEvaluationSpec
) -> TuningEvaluationSpec:
    classifier, parameter, value = request.param
    metrics = ["roc_auc", "average_precision", "balanced_accuracy"]
    if hasattr(classifier, "predict_proba"):
        metrics.append("neg_log_loss")

    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    inner = TuningSpec(
        pipeline=Pipeline([("clf", classifier)]),
        param_distributions={f"clf__{parameter}": CategoricalDistribution([value])},
        data_path=evaluation_spec.inner.data_path,
        n_searches=1,
        cv=inner_cv,
        scoring=metrics,
    )
    return TuningEvaluationSpec(inner=inner, outer_cv=evaluation_spec.outer_cv)
