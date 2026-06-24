from pathlib import Path

import yaml

from covid.data import DataCleaningConfig
from covid.run_config import RunConfig
from covid.training import TrainingConfig


def from_yaml(path: Path) -> RunConfig:
    with open(path, "r") as file:
        config_data = yaml.safe_load(file)
        dataset_config = config_data["dataset"]
    return RunConfig(
        raw_data_path=Path(dataset_config["raw_path"]),
        cleaning=cleaning_config_from_yaml(path),
        train=training_config_from_yaml(path),
    )


def cleaning_config_from_yaml(path: Path) -> DataCleaningConfig:
    with open(path, "r") as file:
        config_data = yaml.safe_load(file)
    config = config_data["dataset"]

    return DataCleaningConfig(
        target_column=config["target_column"],
        id_column=config["id_column"],
        sparse_threshold=config["sparse_threshold"],
    )


def training_config_from_yaml(path: Path) -> TrainingConfig:
    with open(path, "r") as file:
        config_data = yaml.safe_load(file)
    config = config_data["training"]

    return TrainingConfig(
        test_size=config["test_size"],
        random_state=config["random_state"],
        n_folds=config["n_folds"],
        shuffle=config["shuffle"],
        metrics=config["metrics"],
    )
