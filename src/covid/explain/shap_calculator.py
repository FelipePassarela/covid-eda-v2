from typing import cast

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shap import Explanation
from sklearn.base import BaseEstimator


def create_shap_explanation(
    X_background: pd.DataFrame, X_foreground: pd.DataFrame, classifier: BaseEstimator
) -> Explanation:
    masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
    explainer = shap.Explainer(classifier, masker=masker)
    explanation = explainer(X_foreground)
    return cast(Explanation, explanation)


def extract_background_from(
    X_foreground: pd.DataFrame, n_samples: int | None = None
) -> pd.DataFrame:
    if n_samples is None:
        return X_foreground
    return shap.sample(X_foreground, nsamples=n_samples)


def calculate_shap_importances(explanation: Explanation) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": explanation.feature_names,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_shap_explanation(explanation: Explanation, max_display: int = 25) -> Figure:
    fig = plt.figure()
    shap.plots.beeswarm(explanation, show=False, max_display=max_display)
    fig.tight_layout()
    return fig
