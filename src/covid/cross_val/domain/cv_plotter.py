from abc import ABC, abstractmethod

from matplotlib.figure import Figure

from .cv_result import CVResults


class CVPlotter(ABC):
    @abstractmethod
    def plot_results(self, cv_results: CVResults) -> Figure:
        pass

    @abstractmethod
    def figure_filename(self) -> str:
        pass
