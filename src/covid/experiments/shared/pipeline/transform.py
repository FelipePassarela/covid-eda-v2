from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import TunedThresholdClassifierCV


def split_pipeline(pipeline: Pipeline) -> tuple[Pipeline, BaseEstimator]:
    pipeline = unwrap_threshold_model_if_needed(pipeline)

    # imblearn resamplers cannot be used in inference, so we need to remove them
    transformer_steps = [
        (name, step)
        for name, step in pipeline.steps[:-1]
        if not hasattr(step, "fit_resample")
    ]
    preprocessor = Pipeline(steps=transformer_steps)
    classifier = pipeline[-1]

    return preprocessor, classifier


def unwrap_threshold_model_if_needed(
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
