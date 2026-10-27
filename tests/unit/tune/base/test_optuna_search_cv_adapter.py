from types import SimpleNamespace

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial

from covid.tune.base.result import TuningResult
from covid.tune.common.cv_report import OptunaSearchCVAdapter


def test_adapter_handles_failed_trials_and_preserves_row_alignment() -> None:
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
    estimator = object()
    search = SimpleNamespace(
        trials_=study.trials,
        return_train_score=True,
        best_params_=study.best_params,
        best_score_=study.best_value,
        best_estimator_=estimator,
    )

    result = TuningResult.from_fitted_search(OptunaSearchCVAdapter(search, "roc_auc"))

    assert result.best_estimator is estimator
    assert result.best_score == 0.75
    assert result.best_params == {"classifier__strategy": "uniform"}
    report = result.cv_report
    assert report["params"].tolist() == [trial.params for trial in study.trials]
    assert np.isnan(report.loc[0, "mean_test_roc_auc"])
    assert report.loc[1, "mean_test_roc_auc"] == 0.75
    assert report.loc[1, "std_test_roc_auc"] == 0.05
    assert report.loc[1, "mean_train_roc_auc"] == 0.8
    assert report.loc[1, "std_train_roc_auc"] == 0.02
