from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline

from covid_common.feature import TARGET
from covid_model.eval import evaluate
from covid_model.eval.context import EvaluationContext
from covid_model.eval.result import EvaluationResult
from covid_model.eval.tracker import EvaluationTracker


def test_evaluate_tracks_inputs_and_results_in_order(
    evaluated_tracker: MagicMock, pipeline_path: Path, data_path: Path
) -> None:
    evaluated_tracker.track_spec.assert_called_once_with(pipeline_path, data_path)
    evaluated_tracker.track_context.assert_called_once()
    evaluated_tracker.track_result.assert_called_once()
    tracked_events = [event[0] for event in evaluated_tracker.mock_calls]
    assert tracked_events == ["track_spec", "track_context", "track_result"]


def test_evaluate_loads_data_and_fitted_pipeline(
    evaluated_tracker: MagicMock,
    fitted_pipeline: Pipeline,
    classification_df: pd.DataFrame,
) -> None:
    context: EvaluationContext = evaluated_tracker.track_context.call_args.args[0]
    expected_features = classification_df.drop(columns=TARGET)
    expected_target = classification_df[TARGET]

    pd.testing.assert_frame_equal(context.X, expected_features)
    pd.testing.assert_series_equal(context.y, expected_target)
    np.testing.assert_array_equal(
        context.pipeline.predict(expected_features),
        fitted_pipeline.predict(expected_features),
    )


def test_evaluate_tracks_predictions_and_positive_class_scores(
    evaluated_tracker: MagicMock,
    fitted_pipeline: Pipeline,
    classification_df: pd.DataFrame,
) -> None:
    result: EvaluationResult = evaluated_tracker.track_result.call_args.args[0]
    features = classification_df.drop(columns=TARGET)
    expected_target = classification_df[TARGET]
    expected_predictions = pd.Series(
        fitted_pipeline.predict(features), index=expected_target.index
    )
    expected_scores = pd.Series(
        fitted_pipeline.predict_proba(features)[:, 1], index=expected_target.index
    )

    pd.testing.assert_series_equal(result.y_true, expected_target)
    pd.testing.assert_series_equal(result.y_pred, expected_predictions)
    pd.testing.assert_series_equal(result.y_score, expected_scores)


def test_evaluate_prints_metrics_and_confusion_matrix(
    pipeline_path: Path,
    data_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracker = MagicMock(spec=EvaluationTracker)

    evaluate(pipeline_path, data_path, tracker)

    output = capsys.readouterr().out
    for label in (
        "Evaluation scores:",
        "balanced_accuracy",
        "recall",
        "precision",
        "f1",
        "roc_auc",
        "Confusion matrix:",
        "actual_negative",
        "actual_positive",
        "predicted_negative",
        "predicted_positive",
    ):
        assert label in output


@pytest.mark.parametrize("missing_file", ["pipeline", "data"])
def test_evaluate_propagates_missing_file_errors_without_tracking_results(
    pipeline_path: Path, data_path: Path, missing_file: str
) -> None:
    if missing_file == "pipeline":
        pipeline_path = pipeline_path.with_name("missing.pkl")
    else:
        data_path = data_path.with_name("missing.csv")
    tracker = MagicMock(spec=EvaluationTracker)

    with pytest.raises(FileNotFoundError):
        evaluate(pipeline_path, data_path, tracker)

    assert tracker.mock_calls == [call.track_spec(pipeline_path, data_path)]


def test_evaluate_rejects_data_without_target(
    pipeline_path: Path, data_path: Path, classification_df: pd.DataFrame
) -> None:
    classification_df.drop(columns=TARGET).to_csv(data_path, index=False)
    tracker = MagicMock(spec=EvaluationTracker)

    with pytest.raises(KeyError, match=TARGET):
        evaluate(pipeline_path, data_path, tracker)

    tracker.track_context.assert_not_called()
    tracker.track_result.assert_not_called()
