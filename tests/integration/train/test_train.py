from imblearn.pipeline import Pipeline
from sklearn.model_selection import TunedThresholdClassifierCV

from covid.train import TrainingSpec, fit, tune_threshold
from tests.integration.train.conftest import (
    assert_fit_tracker_called,
    assert_model_threshold_is_tuned,
    assert_pipeline_structure,
    assert_saved_model_is_fitted,
    assert_threshold_tuning_tracker_called,
)


def test_fit_trains_and_saves_model(spec: TrainingSpec) -> None:
    fit(spec)

    saved_model = assert_saved_model_is_fitted(spec.model_output_path)
    assert isinstance(saved_model, Pipeline)
    assert_pipeline_structure(saved_model)


def test_tune_threshold_trains_and_saves_tuned_model(spec: TrainingSpec) -> None:
    tune_threshold(spec, scoring="f1")

    saved_model = assert_saved_model_is_fitted(spec.model_output_path)
    assert isinstance(saved_model, TunedThresholdClassifierCV)
    assert_model_threshold_is_tuned(saved_model)


def test_fit_tracks_training(spec: TrainingSpec) -> None:
    fit(spec)
    assert_fit_tracker_called(spec.tracker, spec.model_output_path)


def test_tune_threshold_tracks_training(spec: TrainingSpec) -> None:
    tune_threshold(spec, scoring="f1")
    assert_threshold_tuning_tracker_called(spec.tracker, spec.model_output_path)
