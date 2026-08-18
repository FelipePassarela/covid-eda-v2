from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import TunedThresholdClassifierCV

from covid.data import load_and_split_data


def load_and_split_pipeline(pipeline_path: Path) -> tuple[Pipeline, BaseEstimator]:
    pipeline = joblib.load(pipeline_path)
    pipeline = _unwrap_threshold_model_if_needed(pipeline)
    preprocessor, classifier = _split_pipeline_components(pipeline)
    return preprocessor, classifier


def _unwrap_threshold_model_if_needed(
    model: TunedThresholdClassifierCV | Pipeline,
) -> Pipeline:
    if isinstance(model, TunedThresholdClassifierCV):
        return model.estimator_
    if isinstance(model, Pipeline):
        return model
    raise TypeError(
        "Expected model to be either a Pipeline or TunedThresholdClassifierCV, "
        f"got {type(model)} instead."
    )


def _split_pipeline_components(
    pipeline: Pipeline,
) -> tuple[Pipeline, BaseEstimator]:
    # imblearn resamplers cannot be used in inference, so we need to remove them
    transformer_steps = [
        (name, step)
        for name, step in pipeline.steps[:-1]
        if not hasattr(step, "fit_resample")
    ]
    preprocessor = Pipeline(steps=transformer_steps)
    classifier = pipeline[-1]
    return preprocessor, classifier


def load_and_transform_features(
    preprocessor: Pipeline, data_path: Path
) -> pd.DataFrame:
    X_train, _ = load_and_split_data(data_path)
    return preprocessor.transform(X_train)
