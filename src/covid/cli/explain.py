from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
import typer
from imblearn.pipeline import Pipeline as ImblearnPipeline
from shap import Explanation
from sklearn.base import BaseEstimator
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline as SklearnPipeline

from covid import constants
from covid.data import load_data, split_features_and_target


def main() -> None:
    typer.run(explain)


def explain(
    pipeline_path: Path,
    train_path: Path = constants.INTERIM_TRAIN_DATA_PATH,
    test_path: Path = constants.INTERIM_TEST_DATA_PATH,
) -> None:
    train_data = load_data(train_path)
    X_train, _ = split_features_and_target(train_data)

    threshold_model = joblib.load(pipeline_path)
    pipeline = unwrap_threshold_model(threshold_model)
    preprocessor, classifier = split_fitted_pipeline(pipeline)

    test_data = load_data(test_path)
    X_test, _ = split_features_and_target(test_data)

    X_test_transformed = preprocessor.transform(X_test)
    X_train_transformed = preprocessor.transform(X_train)

    explanation = create_shap_explanation(
        X_train_transformed, X_test_transformed, classifier
    )
    shap.plots.beeswarm(explanation, show=True, max_display=25)


def unwrap_threshold_model(
    threshold_model: TunedThresholdClassifierCV,
) -> ImblearnPipeline:
    if not isinstance(threshold_model, TunedThresholdClassifierCV):
        raise ValueError(
            "The provided model is not a TunedThresholdClassifierCV instance."
        )
    return threshold_model.estimator_


def split_fitted_pipeline(pipeline: ImblearnPipeline) -> tuple[SklearnPipeline, Any]:
    transformer_steps = [
        (name, step)
        for name, step in pipeline.steps[:-1]
        if not hasattr(step, "fit_resample")
    ]
    preprocessor = SklearnPipeline(steps=transformer_steps)
    classifier = pipeline[-1]
    return preprocessor, classifier


def create_shap_explanation(
    X_train_transformed: pd.DataFrame,
    X_test_transformed: pd.DataFrame,
    classifier: BaseEstimator,
) -> Explanation | list[Explanation]:
    masker = shap.maskers.Independent(
        X_train_transformed, max_samples=len(X_train_transformed)
    )
    explainer = shap.Explainer(classifier, masker=masker)
    return explainer(X_test_transformed)


if __name__ == "__main__":
    main()
