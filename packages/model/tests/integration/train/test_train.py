from pathlib import Path
from unittest.mock import MagicMock, Mock

from hydra import compose, initialize_config_module
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.utils.validation import check_is_fitted

from covid_common.feature import ColumnDropper
from covid_common.pipeline import load_pipeline
from covid_model.train import TrainingSpec, fit, tune_threshold
from covid_model.train.spec_factory import training_spec_from_config
from covid_model.train.tracker import TrainingTracker


def test_training_config_builds_shared_pipeline(data_path: Path) -> None:
    model_output_path = Path("models/test.joblib")
    with initialize_config_module(config_module="covid_model.conf", version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "pipeline=logistic-regression",
                f"output_path={model_output_path.as_posix()}",
            ],
        )
    config.dataset.path = str(data_path)

    spec = training_spec_from_config(config, tracker=Mock(spec=TrainingTracker))

    assert spec.data_path == data_path
    assert spec.model_output_path == model_output_path
    assert isinstance(spec.model.named_steps["column_dropper"], ColumnDropper)
    assert isinstance(spec.model.named_steps["classifier"], LogisticRegression)


def test_fit_trains_and_saves_model(spec: TrainingSpec) -> None:
    fit(spec)

    saved_model = _assert_saved_model_is_fitted(spec.model_output_path)
    assert isinstance(saved_model, Pipeline)
    _assert_pipeline_structure(saved_model)


def test_tune_threshold_trains_and_saves_tuned_model(spec: TrainingSpec) -> None:
    tune_threshold(spec, scoring="f1")

    saved_model = _assert_saved_model_is_fitted(spec.model_output_path)
    assert isinstance(saved_model, TunedThresholdClassifierCV)
    _assert_model_threshold_is_tuned(saved_model)


def test_fit_tracks_training(spec: TrainingSpec) -> None:
    fit(spec)
    _assert_fit_tracker_called(spec.tracker, spec.model_output_path)


def test_tune_threshold_tracks_training(spec: TrainingSpec) -> None:
    tune_threshold(spec, scoring="f1")
    _assert_threshold_tuning_tracker_called(spec.tracker, spec.model_output_path)


def _assert_pipeline_structure(pipeline: Pipeline) -> None:
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)


def _assert_saved_model_is_fitted(
    model_path: Path,
) -> Pipeline | TunedThresholdClassifierCV:
    assert model_path.exists()
    saved_model = load_pipeline(model_path)
    check_is_fitted(saved_model)
    return saved_model


def _assert_model_threshold_is_tuned(model: TunedThresholdClassifierCV) -> None:
    assert 0.0 <= model.best_threshold_ <= 1.0
    assert model.best_score_ is not None
    assert model.cv_results_ is not None


def _assert_fit_tracker_called(tracker: MagicMock, model_path: Path) -> None:
    tracker.track_data.assert_called_once()
    tracker.track_model.assert_called_once_with(model_path)


def _assert_threshold_tuning_tracker_called(
    tracker: MagicMock, model_path: Path
) -> None:
    tracker.track_data.assert_called_once()
    tracker.track_model.assert_called_once_with(model_path)
    tracker.track_threshold_tuning.assert_called_once()
