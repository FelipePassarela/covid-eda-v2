from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from covid_common.paths import REPORTS_DIR
from covid_tuning.eval.visualization.data import BoxplotData
from covid_tuning.eval.visualization.io import create_boxplot_data, save_figure
from covid_tuning.eval.visualization.presenter import (
    format_boxplot_dataframe,
    present_metric,
)


def plot_boxplot(
    cv_results: list[Path],
    metric: str = "test_balanced_accuracy",
    output_path: Path = REPORTS_DIR / "tuning-evaluation" / "boxplot.png",
) -> None:
    boxplot_data = create_boxplot_data(cv_results, metric)
    fig = _create_boxplot_figure(boxplot_data)
    save_figure(fig, output_path)


def _create_boxplot_figure(boxplot_data: BoxplotData) -> Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(
        data=format_boxplot_dataframe(boxplot_data),
        x=boxplot_data.metric,
        y="model",
        hue="model",
        ax=ax,
    )
    sns.despine(offset=10, trim=True)
    ax.set_xlabel(present_metric(boxplot_data.metric))
    ax.set_ylabel("Model")

    ax.set_title(
        f"Nested Cross-Validation Score Distribution ({boxplot_data.n_folds} Outer Folds)"
    )

    fig.tight_layout()

    return fig
