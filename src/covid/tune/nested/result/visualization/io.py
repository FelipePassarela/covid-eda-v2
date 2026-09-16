from pathlib import Path

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from covid.tune.nested.result.visualization.data import BoxplotData


def create_boxplot_data(cv_results_paths: list[Path], metric: str) -> BoxplotData:
    model_names = []
    scores = []

    for cv_path in cv_results_paths:
        df = pd.read_csv(cv_path)
        logger.info("Loaded cv results from {}", cv_path)
        logger.debug(df.head())

        model_names.append(cv_path.stem)
        scores.append(df[metric].tolist())

    return BoxplotData(model_names=model_names, folds_scores=scores, metric=metric)


def save_figure(fig: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    logger.success(f"Figure saved to {output_path}")
    plt.show()
