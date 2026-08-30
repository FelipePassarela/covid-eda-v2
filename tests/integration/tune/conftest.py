from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from covid.tune import TuningResult, TuningSpec


@pytest.fixture()
def spec(tmp_path: Path, classification_df: pd.DataFrame) -> TuningSpec:
    data_path = tmp_path / "data.csv"
    classification_df.to_csv(data_path, index=False)

    classifier = DummyClassifier(strategy="most_frequent")

    return TuningSpec(
        name="test_tune",
        data_path=data_path,
        pipeline=Pipeline(steps=[("clf", classifier)]),
        param_distributions={"clf__strategy": ["most_frequent", "stratified"]},
        n_searches=2,
        n_fold_repeats=1,
        scoring=["accuracy"],
    )


def assert_pipeline_structure(pipeline: Pipeline) -> None:
    assert isinstance(pipeline.named_steps["clf"], DummyClassifier)


def assert_result(result: TuningResult, spec: TuningSpec) -> None:
    check_is_fitted(result.best_estimator)
    assert isinstance(result.best_estimator, Pipeline)
    assert_pipeline_structure(result.best_estimator)

    assert result.best_params["clf__strategy"] in ["most_frequent", "stratified"]
    assert 0.0 <= result.best_score <= 1.0

    assert isinstance(result.report, pd.DataFrame)
    assert len(result.report) == spec.n_searches
    assert f"mean_test_{spec.scoring[0]}" in result.report.columns


def assert_mocks_called(
    tune_fn: MagicMock, spec: MagicMock, tracker: MagicMock, expected_result: MagicMock
) -> None:
    tune_fn.assert_called_once_with(spec)
    tracker.track_spec.assert_called_once_with(spec)
    tracker.track_result.assert_called_once_with(expected_result)
