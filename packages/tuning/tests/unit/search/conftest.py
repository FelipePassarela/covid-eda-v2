from types import SimpleNamespace

import optuna
import pytest
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial


@pytest.fixture()
def search_with_failed_trial() -> SimpleNamespace:
    study = optuna.create_study(direction="maximize")
    distributions = {
        "classifier__strategy": CategoricalDistribution(["prior", "uniform"])
    }
    study.add_trial(
        create_trial(
            params={"classifier__strategy": "prior"},
            distributions=distributions,
            state=TrialState.FAIL,
        )
    )
    study.add_trial(
        create_trial(
            params={"classifier__strategy": "uniform"},
            distributions=distributions,
            value=0.75,
            user_attrs={
                "mean_test_score": 0.75,
                "std_test_score": 0.05,
                "mean_train_score": 0.8,
                "std_train_score": 0.02,
            },
        )
    )
    return SimpleNamespace(
        trials_=study.trials,
        return_train_score=True,
        best_params_=study.best_params,
        best_score_=study.best_value,
        best_estimator_=object(),
    )
