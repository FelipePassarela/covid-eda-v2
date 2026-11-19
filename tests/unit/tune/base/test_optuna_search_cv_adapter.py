from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

from covid.tune.common.cv_report import OptunaSearchCVAdapter
from covid.tune.search.result import TuningResult


def test_adapter_handles_failed_trials_and_preserves_row_alignment(
    search_with_failed_trial: SimpleNamespace,
) -> None:
    search_adapter = OptunaSearchCVAdapter(search_with_failed_trial, "roc_auc")
    result = TuningResult.from_fitted_search(search_adapter)

    assert result.best_estimator is search_with_failed_trial.best_estimator_
    assert result.best_score == 0.75
    assert result.best_params == {"classifier__strategy": "uniform"}
    expected_report = pd.DataFrame(
        {
            "params": [
                {"classifier__strategy": "prior"},
                {"classifier__strategy": "uniform"},
            ],
            "mean_test_roc_auc": [float("nan"), 0.75],
            "std_test_roc_auc": [float("nan"), 0.05],
            "mean_train_roc_auc": [float("nan"), 0.8],
            "std_train_roc_auc": [float("nan"), 0.02],
        }
    )
    assert_frame_equal(result.cv_report, expected_report)
