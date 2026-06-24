from abc import ABC, abstractmethod

from matplotlib.figure import Figure

from .cv_result import CVResults


class CVTracker(ABC):
    @abstractmethod
    def log_cv_results(self, cv_results: CVResults) -> None:
        """Log cross-validation results."""
        pass

    @abstractmethod
    def log_model_params(self, model_name: str, params: dict) -> None:
        pass

    @abstractmethod
    def log_cv_plot(self, fig: Figure, filename: str = "plot.png") -> None:
        pass
