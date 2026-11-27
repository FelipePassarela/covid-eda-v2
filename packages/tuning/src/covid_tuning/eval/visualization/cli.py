import typer

from covid_tuning.eval.visualization.boxplot import plot_boxplot


def main() -> None:
    typer.run(plot_boxplot)
