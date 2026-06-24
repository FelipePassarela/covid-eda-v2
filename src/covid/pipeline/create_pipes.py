from sklearn.pipeline import Pipeline

from covid.feature import (
    IMPUTER_REGISTRY,
    REDUCER_REGISTRY,
    SCALER_REGISTRY,
    PreprocessorFactory,
)
from covid.model import MODEL_ALIASES, MODEL_REGISTRY, ModelFactory


def create_pipes(random_state: int | None) -> dict[str, Pipeline]:
    model_factory = ModelFactory(MODEL_REGISTRY, MODEL_ALIASES)
    preprocessor_factory = PreprocessorFactory(
        IMPUTER_REGISTRY, SCALER_REGISTRY, REDUCER_REGISTRY
    )

    # TODO: this is a bit hacky - we should have a more flexible way to define pipelines in the config
    linear_preprocessor = Pipeline(
        [
            ("imputer", preprocessor_factory.create_imputer("simple")),
            ("scaler", preprocessor_factory.create_scaler("standard")),
            (
                "reducer",
                preprocessor_factory.create_reducer(
                    "pca", n_components=0.95, random_state=random_state
                ),
            ),
        ]
    )
    tree_preprocessor = Pipeline(
        [
            ("imputer", preprocessor_factory.create_imputer("simple")),
            (
                "reducer",
                preprocessor_factory.create_reducer(
                    "pca", n_components=0.95, random_state=random_state
                ),
            ),
        ]
    )

    lr_pipe = Pipeline(
        [
            ("preprocessor", linear_preprocessor),
            ("model", model_factory.create("lr", random_state=random_state)),
        ]
    )
    svc_pipe = Pipeline(
        [
            ("preprocessor", linear_preprocessor),
            ("model", model_factory.create("svc", random_state=random_state)),
        ]
    )
    xgb_pipe = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            ("model", model_factory.create("xgb", random_state=random_state)),
        ]
    )
    rf_pipe = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            ("model", model_factory.create("rf", random_state=random_state)),
        ]
    )
    dmy_pipe = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "model",
                model_factory.create(
                    "dmy", strategy="most_frequent", random_state=random_state
                ),
            ),
        ]
    )

    models = {
        "lr": lr_pipe,
        "svc": svc_pipe,
        "xgb": xgb_pipe,
        "rf": rf_pipe,
        "dmy": dmy_pipe,
    }

    return models
