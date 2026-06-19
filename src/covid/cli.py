from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.pipeline import Pipeline

from covid import config_loader
from covid.app_config import AppConfig
from covid.data import CovidDataset
from covid.feature import (
    IMPUTER_REGISTRY,
    REDUCER_REGISTRY,
    SCALER_REGISTRY,
    PreprocessorFactory,
)
from covid.metric import METRIC_REGISTRY, MetricFactory
from covid.model import MODEL_ALIASES, MODEL_REGISTRY, ModelFactory
from covid.training import CrossValidator

app = typer.Typer()


@app.command()
def cli(config_path: Path = Path("config/config.yaml")) -> None:
    app_config = config_loader.from_yaml(config_path)
    run_experiment(app_config)


def run_experiment(app_config: AppConfig) -> None:
    logger.info(f"Loading dataset from {app_config.dataset.raw_path}")
    raw_df = pd.read_csv(app_config.dataset.raw_path)  # TODO: abstract to DatasetLoader

    X, y = (
        CovidDataset.from_dataframe(
            raw_df,
            target=app_config.dataset.target_column,
            id_column=app_config.dataset.id_column,
        )
        .without_sparse_columns(threshold=app_config.dataset.sparse_threshold)
        .as_categorical()
        .split()
    )
    logger.info("Processed X shape: {shape}", shape=X.shape)
    logger.info("Processed y shape: {shape}", shape=y.shape)

    random_state = app_config.train.random_state
    pipes = create_pipes(random_state)
    metrics = MetricFactory(METRIC_REGISTRY).create_all(app_config.train.metrics)

    logger.info("Created {n_models} models", n_models=len(pipes))
    cv = CrossValidator(
        n_folds=app_config.train.n_folds,
        metrics=metrics,
        random_state=random_state,
        shuffle=app_config.train.shuffle,
    )
    result = cv.run(pipes, X, y)

    sort_by = app_config.train.metrics[0]
    summary = result.summarize(sort_by=sort_by)
    logger.info("Cross-validation summary:\n{summary}", summary=summary)


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


if __name__ == "__main__":
    app()
