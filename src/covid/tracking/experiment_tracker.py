from abc import ABC, abstractmethod

from covid.run_config import RunConfig


class ExperimentTracker(ABC):
    @abstractmethod
    def log_run_config(self, config: RunConfig) -> None:
        pass
