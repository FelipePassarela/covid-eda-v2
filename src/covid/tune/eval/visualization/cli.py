import typer

from covid.tune.eval.visualization.boxplot import plot_boxplot


def main() -> None:
    typer.run(plot_boxplot)
