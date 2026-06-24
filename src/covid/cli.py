from datetime import datetime
from pathlib import Path

import typer

from covid import config_loader
from covid.experiment import run_experiment
from covid.tracking import (
    FileExperimentTracker,
)
from covid.training import FileCVTracker

app = typer.Typer()


@app.command()
def cli(config_path: Path = Path("config/config.yaml")) -> None:
    run_config = config_loader.from_yaml(config_path)

    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    tracker = FileExperimentTracker(log_dir=Path(f"logs/run_{now}"))
    cv_tracker = FileCVTracker(output_dir=Path(f"logs/run_{now}/cv"))
    
    run_experiment(run_config, tracker, cv_tracker)


if __name__ == "__main__":
    app()
