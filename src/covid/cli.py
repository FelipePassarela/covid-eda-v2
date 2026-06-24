from datetime import datetime
from pathlib import Path

import typer

from covid import config_loader
from covid.cross_val import BoxplotCVPlotter, FileCVTracker
from covid.data import CovidDatasetCSVLoader, FileCovidDatasetTracker
from covid.experiment import run_experiment

app = typer.Typer()


@app.command()
def cli(config_path: Path = Path("config/config.yaml")) -> None:
    run_config = config_loader.from_yaml(config_path)

    now = datetime.now().strftime("%d%m%Y_%H%M%S")

    dataset_loader = CovidDatasetCSVLoader(raw_path=run_config.raw_data_path)
    dataset_tracker = FileCovidDatasetTracker(
        output_dir=Path(f"logs/run_{now}/dataset")
    )
    cv_tracker = FileCVTracker(output_dir=Path(f"logs/run_{now}/cv"))
    cv_plotters = [BoxplotCVPlotter(evaluation_metric=run_config.train.metrics[0])]

    run_experiment(
        run_config=run_config,
        covid_dataset_loader=dataset_loader,
        dataset_tracker=dataset_tracker,
        cv_tracker=cv_tracker,
        cv_plotters=cv_plotters,
    )


if __name__ == "__main__":
    app()
