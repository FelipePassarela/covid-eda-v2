from matplotlib import pyplot as plt

from covid.experiments.explain.result import ExplainingResult
from covid.experiments.explain.shap_calculator import (
    calculate_shap_importances,
    create_shap_explanation,
    plot_shap_explanation,
)
from covid.experiments.explain.spec import ExplainingSpec
from covid.experiments.explain.tracker import ExplainingTracker
from covid.experiments.shared.pipeline import (
    load_and_transform_features,
    load_pipeline,
    split_pipeline,
)


def explain(spec: ExplainingSpec, tracker: ExplainingTracker) -> None:
    tracker.track_spec(spec)
    result = _explain(spec)
    tracker.track_result(result)
    plt.close(result.beeswarm_plot)


def _explain(spec: ExplainingSpec) -> ExplainingResult:
    pipeline = load_pipeline(spec.pipeline_path)
    preprocessor, classifier = split_pipeline(pipeline)

    X_train_transformed = load_and_transform_features(preprocessor, spec.train_path)
    X_test_transformed = load_and_transform_features(preprocessor, spec.test_path)

    explanation = create_shap_explanation(
        X_train_transformed, X_test_transformed, classifier
    )
    importances = calculate_shap_importances(explanation)

    beeswarm_plot = plot_shap_explanation(explanation, max_display=spec.max_display)

    return ExplainingResult(
        X_test_transformed=X_test_transformed,
        X_train_transformed=X_train_transformed,
        pipeline=pipeline,
        importances=importances,
        explanation=explanation,
        beeswarm_plot=beeswarm_plot,
    )
