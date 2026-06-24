from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from covid.run_config import RunConfig
from covid.tracking.experiment_tracker import ExperimentTracker


class FileExperimentTracker(ExperimentTracker):
    def __init__(self, log_dir: Path) -> None:
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.log_dir = log_dir / f"run_{now}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_run_config(self, config: RunConfig) -> None:
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(config), f)
