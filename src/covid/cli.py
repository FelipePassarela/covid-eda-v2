from datetime import datetime
from pathlib import Path

import sklearn
import typer

from covid import commands, config_loader
from covid.cross_val import BoxplotCVPlotter, FileCVTracker
from covid.data import (
    CovidDatasetCSVLoader,
    FileCovidDatasetTracker,
)

app = typer.Typer()


@app.command()
def cross_validation(config_path: Path = Path("config/config.yaml")) -> None:
    run_config = config_loader.from_yaml(config_path)

    now = datetime.now().strftime("%d%m%Y_%H%M%S")

    dataset_loader = CovidDatasetCSVLoader(raw_path=run_config.raw_data_path)
    dataset_tracker = FileCovidDatasetTracker(
        output_dir=Path(f"logs/run_{now}/dataset")
    )
    cv_tracker = FileCVTracker(output_dir=Path(f"logs/run_{now}/cv"))
    cv_plotters = [BoxplotCVPlotter(evaluation_metric=run_config.training.metrics[0])]

    commands.run_cross_validation(
        run_config=run_config,
        covid_dataset_loader=dataset_loader,
        dataset_tracker=dataset_tracker,
        cv_tracker=cv_tracker,
        cv_plotters=cv_plotters,
    )


@app.command()
def feature_selection(config_path: Path = Path("config/config.yaml")) -> None:
    run_config = config_loader.from_yaml(config_path)
    dataset_loader = CovidDatasetCSVLoader(raw_path=run_config.raw_data_path)
    commands.run_feature_selection(
        n_features=50,  # TODO: Make selector params configurable
        train_config=run_config.training,
        cleaning_config=run_config.cleaning,
        dataset_loader=dataset_loader,
        output_path=Path("data/processed"),
    )


@app.command()
def plot_shap(config_path: Path = Path("config/config.yaml")) -> None:
    run_config = config_loader.from_yaml(config_path)
    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    dataset_loader = CovidDatasetCSVLoader(raw_path=run_config.raw_data_path)

    commands.generate_shap_plots(
        dataset_loader=dataset_loader,
        training_config=run_config.training,
        cleaning_config=run_config.cleaning,
        output_path=Path(f"logs/run_{now}/shap"),
    )


if __name__ == "__main__":
    sklearn.set_config(transform_output="pandas")
    app()
