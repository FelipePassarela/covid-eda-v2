import typer

from covid.tune.nested.result.visualization.boxplot import plot_boxplot


def main() -> None:
    typer.run(plot_boxplot)
