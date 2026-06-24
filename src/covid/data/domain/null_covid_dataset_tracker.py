from pathlib import Path

from covid.data.domain.covid_dataset import CovidDataset, DataCleaningConfig

from .covid_dataset_tracker import CovidDatasetTracker


class NullCovidDatasetTracker(CovidDatasetTracker):
    """A tracker that does nothing. Useful when tracking is not needed."""

    def log_cleaning_config(self, cleaning_config: DataCleaningConfig) -> None:
        pass

    def log_dataset_metrics(self, dataset: CovidDataset) -> None:
        pass

    def log_data_path(self, path: Path) -> None:
        pass
