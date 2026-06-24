import pandas as pd
from loguru import logger

from covid.data import CovidDataset
from covid.metric import METRIC_REGISTRY, MetricFactory
from covid.pipeline import create_pipes
from covid.run_config import RunConfig
from covid.tracking import ExperimentTracker, NullExperimentTracker
from covid.training import CrossValidator, CVTracker
from covid.training.null_cv_tracker import NullCVTracker


def run_experiment(
    run_config: RunConfig,
    tracker: ExperimentTracker = NullExperimentTracker(),
    cv_tracker: CVTracker = NullCVTracker(),
) -> None:
    tracker.log_run_config(run_config)
    logger.info(f"Loading dataset from {run_config.dataset.raw_path}")
    raw_df = pd.read_csv(run_config.dataset.raw_path)  # TODO: abstract to DatasetLoader

    X, y = (
        CovidDataset.from_dataframe(
            raw_df,
            target=run_config.dataset.target_column,
            id_column=run_config.dataset.id_column,
        )
        .without_sparse_columns(threshold=run_config.dataset.sparse_threshold)
        .as_categorical()
        .split()
    )
    logger.info("Processed X shape: {shape}", shape=X.shape)
    logger.info("Processed y shape: {shape}", shape=y.shape)

    random_state = run_config.train.random_state
    pipes = create_pipes.create_pipes(random_state)
    metrics = MetricFactory(METRIC_REGISTRY).create_all(run_config.train.metrics)

    logger.info("Created {n_models} models", n_models=len(pipes))
    cv = CrossValidator(
        n_folds=run_config.train.n_folds,
        metrics=metrics,
        random_state=random_state,
        shuffle=run_config.train.shuffle,
        cv_tracker=cv_tracker,
    )
    result = cv.run(pipes, X, y)

    sort_by = run_config.train.metrics[0]
    summary = result.summarize(sort_by=sort_by)
    logger.info("Cross-validation summary:\n{summary}", summary=summary)
