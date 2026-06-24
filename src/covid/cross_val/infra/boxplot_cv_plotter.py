import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from covid.model import MODEL_ALIASES

from ..domain.cv_plotter import CVPlotter
from ..domain.cv_result import CVResults


class BoxplotCVPlotter(CVPlotter):
    def __init__(self, evaluation_metric: str):
        self._eval_metric = evaluation_metric
        sns.set_style("whitegrid")

    def plot_results(self, cv_results: CVResults) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 6))

        results_df = cv_results.to_dataframe()
        results_df = results_df.sort_values(by=self._eval_metric, ascending=False)
        results_df["model"] = results_df["model"].apply(
            lambda x: MODEL_ALIASES.get(x)
        )  # Use alias if available

        sns.boxplot(
            data=results_df,
            ax=ax,
            x="model",
            y=self._eval_metric,
            hue="model",
            dodge=False,
        )

        kfolds = results_df["fold"].nunique()
        ax.set_title(f"Cross-Validation Results ({kfolds} Folds)")
        ax.set_xlabel("Models")
        ax.set_ylabel(self._eval_metric)

        return fig

    def figure_filename(self) -> str:
        return f"boxplot_{self._eval_metric}.png"
