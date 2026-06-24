from loguru import logger

from covid.data import CovidDatasetLoader, CovidDatasetTracker, NullCovidDatasetTracker
from covid.metric import METRIC_REGISTRY, MetricFactory
from covid.pipeline import create_pipes
from covid.run_config import RunConfig
from covid.cross_val import CrossValidator, CVTracker, NullCVTracker


def run_experiment(
    run_config: RunConfig,
    covid_dataset_loader: CovidDatasetLoader,
    dataset_tracker: CovidDatasetTracker = NullCovidDatasetTracker(),
    cv_tracker: CVTracker = NullCVTracker(),
) -> None:
    dataset_tracker.log_data_path(run_config.raw_data_path)
    dataset_tracker.log_cleaning_config(run_config.cleaning)

    dataset = covid_dataset_loader.load_dataset(run_config.cleaning)
    dataset_tracker.log_dataset_metrics(dataset)

    random_state = run_config.train.random_state
    pipes = create_pipes.create_pipes(random_state)
    metrics = MetricFactory(METRIC_REGISTRY).create_all(run_config.train.metrics)

    logger.info("Created {n_models} models", n_models=len(pipes))

    X, y = dataset.split()
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
