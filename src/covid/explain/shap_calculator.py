from typing import cast

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shap import Explanation
from sklearn.base import BaseEstimator


def create_shap_explanation(
    X_train_transformed: pd.DataFrame,
    X_test_transformed: pd.DataFrame,
    classifier: BaseEstimator,
) -> Explanation:
    masker = shap.maskers.Independent(
        X_train_transformed, max_samples=len(X_train_transformed)
    )
    explainer = shap.Explainer(classifier, masker=masker)
    explanation = explainer(X_test_transformed)
    return cast(Explanation, explanation)


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
    return fig
