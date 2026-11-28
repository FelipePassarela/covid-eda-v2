from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from covid_common.feature import TARGET
from covid_model.explain import ExplainingSpec, explain
from covid_model.explain.result import ExplainingResult
from covid_model.explain.tracker import ExplainingTracker


def assert_plot_is_open(result: ExplainingResult) -> None:
    assert plt.fignum_exists(result.beeswarm_plot.number)


def test_explain_tracks_result_before_closing_plot(
    explaining_spec: ExplainingSpec, existing_figures: set[int]
) -> None:
    tracker = MagicMock(spec=ExplainingTracker)
    tracker.track_result.side_effect = assert_plot_is_open

    explain(explaining_spec, tracker)

    tracker.track_spec.assert_called_once_with(explaining_spec)
    tracker.track_result.assert_called_once()
    result = tracker.track_result.call_args.args[0]
    displayed_features = result.beeswarm_plot.axes[0].get_yticklabels()
    assert len(displayed_features) == explaining_spec.max_display
    assert set(plt.get_fignums()) == existing_figures


def test_explain_transforms_features_and_preserves_pipeline_predictions(
    explaining_result: ExplainingResult,
    fitted_pipeline: Pipeline,
    classification_df: pd.DataFrame,
) -> None:
    X = classification_df.drop(columns=TARGET)
    expected_foreground = fitted_pipeline[:-1].transform(X)

    pd.testing.assert_frame_equal(explaining_result.X_foreground, expected_foreground)
    pd.testing.assert_frame_equal(explaining_result.X_background, expected_foreground)
    np.testing.assert_array_equal(
        explaining_result.pipeline.predict(X), fitted_pipeline.predict(X)
    )


def test_explain_produces_shap_values_matching_classifier_predictions(
    explaining_result: ExplainingResult, fitted_pipeline: Pipeline
) -> None:
    explanation = explaining_result.explanation
    foreground = explaining_result.X_foreground

    assert list(explanation.feature_names) == list(foreground.columns)
    assert explanation.values.shape == foreground.shape
    assert np.isfinite(explanation.values).all()
    np.testing.assert_allclose(explanation.data, foreground)

    shap_log_odds = explanation.base_values + explanation.values.sum(axis=1)
    classifier_log_odds = fitted_pipeline[-1].decision_function(foreground)
    np.testing.assert_allclose(shap_log_odds, classifier_log_odds, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "pipeline_classifier",
    [
        pytest.param(
            SVC(kernel="linear", probability=True, random_state=42), id="linear"
        ),
        pytest.param(SVC(kernel="rbf", probability=True, random_state=42), id="rbf"),
    ],
)
def test_explain_svc_attributes_positive_class_probability(
    explaining_result: ExplainingResult, fitted_pipeline: Pipeline
) -> None:
    explanation = explaining_result.explanation
    foreground = explaining_result.X_foreground
    classifier = fitted_pipeline[-1]
    positive_class_index = list(classifier.classes_).index(1)
    class_probabilities = classifier.predict_proba(foreground)
    expected_probabilities = class_probabilities[:, positive_class_index]

    assert explanation.values.shape == foreground.shape
    shap_probabilities = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(shap_probabilities, expected_probabilities)


def test_explain_sorts_feature_importances_by_magnitude(
    explaining_result: ExplainingResult,
) -> None:
    importances = explaining_result.importances
    mean_abs_shap = importances["mean_abs_shap"]
    expected_features = set(explaining_result.X_foreground.columns)

    assert set(importances["feature"]) == expected_features
    assert mean_abs_shap.ge(0).all()
    assert mean_abs_shap.is_monotonic_decreasing


@pytest.mark.parametrize("missing_file", ["pipeline", "data"])
def test_explain_propagates_missing_file_errors_without_tracking_results(
    pipeline_path: Path,
    data_path: Path,
    missing_file: str,
    existing_figures: set[int],
) -> None:
    if missing_file == "pipeline":
        pipeline_path = pipeline_path.with_name("missing.pkl")
    else:
        data_path = data_path.with_name("missing.csv")
    spec = ExplainingSpec(pipeline_path, data_path)
    tracker = MagicMock(spec=ExplainingTracker)

    with pytest.raises(FileNotFoundError):
        explain(spec, tracker)

    tracker.track_spec.assert_called_once_with(spec)
    tracker.track_result.assert_not_called()
    assert set(plt.get_fignums()) == existing_figures
