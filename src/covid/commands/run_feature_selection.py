from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from covid.cross_val import TrainingConfig
from covid.data import CovidDatasetLoader, DataCleaningConfig
from covid.feature import (
    PREPROCESSOR_REGISTRY,
    PreprocessorFactory,
    select_best_features,
)
from covid.model import MODEL_ALIASES, MODEL_REGISTRY, ModelFactory


def run_feature_selection(
    n_features: int,
    train_config: TrainingConfig,
    cleaning_config: DataCleaningConfig,
    dataset_loader: CovidDatasetLoader,
    output_path: Path,
) -> None:
    logger.info(
        "Running feature selection with {n_features} features to select",
        n_features=n_features,
    )

    dataset = dataset_loader.load_dataset(cleaning_config)
    X, y = dataset.split()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_config.test_size,
        random_state=train_config.random_state,
        stratify=y,
    )

    prep_factory = PreprocessorFactory(PREPROCESSOR_REGISTRY)
    imputer = prep_factory.create_imputer("simple", strategy="most_frequent")
    scaler = prep_factory.create_scaler("standard")

    rfe_estimator = ModelFactory(MODEL_REGISTRY, MODEL_ALIASES).create(
        "svc",
        random_state=train_config.random_state,
        probability=True,
        kernel="linear",
    )
    selector = prep_factory.create_selector(  # TODO: Make selector params configurable
        "rfe",
        estimator=rfe_estimator,
        n_features_to_select=n_features,
        step=1,
        verbose=1,
    )
    selector_pipe = Pipeline(
        [("imputer", imputer), ("scaler", scaler), ("rfe", selector)]
    )

    X_train, X_test = select_best_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selector=selector_pipe,
        allow_data_leakage=True,  # TODO: Make this configurable
    )

    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    data = pd.concat([X_full, y_full], axis=1)

    # TODO: Create abstracted data saving
    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_path = output_path / f"selected_{n_features}_{now}.csv"
    data.to_csv(output_path, index=False)
    logger.success(
        "Feature selection completed. Selected features saved to {output_path}",
        output_path=output_path,
    )
