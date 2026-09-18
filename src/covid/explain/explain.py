from covid.common.pipeline import (
    load_and_transform_features,
    load_pipeline,
    split_pipeline,
)
from covid.explain.result import ExplainingResult
from covid.explain.shap_calculator import (
    calculate_shap_importances,
    create_shap_explanation,
    extract_background_from,
    plot_shap_explanation,
)
from covid.explain.spec import ExplainingSpec
from covid.explain.tracker import ExplainingTracker


def explain(spec: ExplainingSpec, tracker: ExplainingTracker) -> None:
    tracker.track_spec(spec)
    result = _explain(spec)
    tracker.track_result(result)
    result.close_plot()


def _explain(spec: ExplainingSpec) -> ExplainingResult:
    pipeline = load_pipeline(spec.pipeline_path)
    preprocessor, classifier = split_pipeline(pipeline)

    X_foreground = load_and_transform_features(spec.data_path, preprocessor)
    X_background = extract_background_from(X_foreground)

    explanation = create_shap_explanation(X_background, X_foreground, classifier)
    importances = calculate_shap_importances(explanation)

    beeswarm_plot = plot_shap_explanation(explanation, max_display=spec.max_display)

    return ExplainingResult(
        X_foreground=X_foreground,
        X_background=X_background,
        pipeline=pipeline,
        importances=importances,
        explanation=explanation,
        beeswarm_plot=beeswarm_plot,
    )
