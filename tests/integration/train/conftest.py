from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.utils.validation import check_is_fitted

from covid.common.pipeline import load_pipeline
from covid.train import TrainingSpec


@pytest.fixture()
def spec(tmp_path: Path, classification_df: pd.DataFrame) -> TrainingSpec:
    data_path = tmp_path / "data.csv"
    classification_df.to_csv(data_path, index=False)

    classifier = LogisticRegression(solver="saga", random_state=42)

    return TrainingSpec(
        model=Pipeline(steps=[("clf", classifier)]),
        data_path=data_path,
        model_output_path=tmp_path / "model.pkl",
        tracker=MagicMock(),
    )


def assert_pipeline_structure(pipeline: Pipeline) -> None:
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)


def assert_saved_model_is_fitted(
    model_path: Path,
) -> Pipeline | TunedThresholdClassifierCV:
    assert model_path.exists()
    saved_model = load_pipeline(model_path)
    check_is_fitted(saved_model)
    return saved_model


def assert_model_threshold_is_tuned(model: TunedThresholdClassifierCV) -> None:
    assert 0.0 <= model.best_threshold_ <= 1.0
    assert model.best_score_ is not None
    assert model.cv_results_ is not None


def assert_fit_tracker_called(tracker: MagicMock, model_path: Path) -> None:
    tracker.track_data.assert_called_once()
    tracker.track_model.assert_called_once_with(model_path)


def assert_threshold_tuning_tracker_called(
    tracker: MagicMock, model_path: Path
) -> None:
    tracker.track_data.assert_called_once()
    tracker.track_model.assert_called_once_with(model_path)
    tracker.track_threshold_tuning.assert_called_once()
