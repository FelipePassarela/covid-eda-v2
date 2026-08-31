from dataclasses import dataclass

import pandas as pd
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shap import Explanation


@dataclass(frozen=True)
class ExplainingResult:
    X_test_transformed: pd.DataFrame
    X_train_transformed: pd.DataFrame
    pipeline: Pipeline
    explanation: Explanation
    importances: pd.DataFrame
    beeswarm_plot: Figure

    def close_plot(self) -> None:
        plt.close(self.beeswarm_plot)
