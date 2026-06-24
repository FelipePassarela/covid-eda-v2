from loguru import logger

from covid.data import CovidDatasetLoader
from covid.metric import METRIC_REGISTRY, MetricFactory
from covid.pipeline import create_pipes
from covid.run_config import RunConfig
from covid.tracking import ExperimentTracker, NullExperimentTracker
from covid.training import CrossValidator, CVTracker
from covid.training.null_cv_tracker import NullCVTracker


def run_experiment(
    run_config: RunConfig,
    covid_dataset_loader: CovidDatasetLoader,
    tracker: ExperimentTracker = NullExperimentTracker(),
    cv_tracker: CVTracker = NullCVTracker(),
) -> None:
    tracker.log_run_config(run_config)

    X, y = covid_dataset_loader.load_dataset(run_config.cleaning).split()

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
