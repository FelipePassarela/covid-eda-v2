from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from covid.data import CovidDataset

from ..domain.covid_dataset import DataCleaningConfig
from ..domain.covid_dataset_tracker import CovidDatasetTracker


class FileCovidDatasetTracker(CovidDatasetTracker):
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def log_cleaning_config(self, cleaning_config: DataCleaningConfig) -> None:
        config_path = self._output_dir / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "a") as f:
            yaml.dump(asdict(cleaning_config), f, default_flow_style=False)

    def log_dataset_metrics(self, dataset: CovidDataset) -> None:
        metrics_path = self._output_dir / "metrics.yaml"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = self.compute_metrics(dataset)

        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False)

        head_tail_path = self._output_dir / "head_tail.csv"
        head_tail_path.parent.mkdir(parents=True, exist_ok=True)
        head_tail = pd.concat([dataset.X.head(), dataset.X.tail()])

        with open(head_tail_path, "w") as f:
            head_tail.to_csv(f, index=False)

    def log_data_path(self, path: Path) -> None:
        # TODO: move this constant to constants.py
        path_log_path = self._output_dir / "config.yaml"
        path_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(path_log_path, "a") as f:
            yaml.dump({"data_path": str(path)}, f, default_flow_style=False)
