from matplotlib.figure import Figure

from .cv_result import CVResults
from .cv_tracker import CVTracker


class NullCVTracker(CVTracker):
    """
    A CVTracker implementation that does nothing.
    Useful for testing or when tracking is not needed.
    """

    def log_cv_results(self, cv_results: CVResults) -> None:
        pass

    def log_model_params(self, model_name: str, params: dict) -> None:
        pass

    def log_cv_plot(self, fig: Figure, filename: str = "plot.png") -> None:
        pass
