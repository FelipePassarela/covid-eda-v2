from pathlib import Path

import pandas as pd
import shap
from shap import Explanation
from sklearn.base import BaseEstimator

from covid.pipeline import load_and_split_pipeline, load_and_transform_features


def explain(pipeline_path: Path, train_path: Path, test_path: Path) -> None:
    preprocessor, classifier = load_and_split_pipeline(pipeline_path)

    X_train_transformed = load_and_transform_features(preprocessor, train_path)
    X_test_transformed = load_and_transform_features(preprocessor, test_path)

    explanation = _create_shap_explanation(
        X_train_transformed, X_test_transformed, classifier
    )
    _plot_shap_explanation(explanation)


def _create_shap_explanation(
    X_train_transformed: pd.DataFrame,
    X_test_transformed: pd.DataFrame,
    classifier: BaseEstimator,
) -> Explanation:
    masker = shap.maskers.Independent(
        X_train_transformed, max_samples=len(X_train_transformed)
    )
    explainer = shap.Explainer(classifier, masker=masker)
    return explainer(X_test_transformed)


def _plot_shap_explanation(explanation: Explanation) -> None:
    shap.plots.beeswarm(explanation, show=True, max_display=25)
