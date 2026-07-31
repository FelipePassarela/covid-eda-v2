from pathlib import Path
from typing import Any

import joblib
import shap
import typer
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline as SklearnPipeline

from covid import constants
from covid.data import load_data, split_features_and_target


def main() -> None:
    typer.run(explain)


def explain(
    pipeline_path: Path, data_path: Path = constants.INTERIM_TRAIN_DATA_PATH
) -> None:
    data = load_data(data_path)
    X, y = split_features_and_target(data)

    threshold_model = joblib.load(pipeline_path)
    pipeline = unwrap_threshold_model(threshold_model)
    preprocessor, classifier = split_fitted_pipeline(pipeline)

    X_transformed = preprocessor.transform(X)

    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(X_transformed)
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


if __name__ == "__main__":
    main()
