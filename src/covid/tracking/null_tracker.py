from covid.run_config import RunConfig
from covid.tracking.experiment_tracker import ExperimentTracker


class NullExperimentTracker(ExperimentTracker):
    """A tracker that does nothing. Useful when tracking is not needed."""

    def log_run_config(self, config: RunConfig) -> None:
        pass
