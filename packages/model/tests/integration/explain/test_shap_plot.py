from unittest.mock import patch

import pytest
import shap
from matplotlib import pyplot as plt
from shap import Explanation

from covid_model.explain.shap_calculator import plot_shap_explanation


@pytest.mark.parametrize("max_display", [2, 25])
def test_plot_creates_figure_without_displaying_it(
    explanation: Explanation, max_display: int, existing_figures: set[int]
) -> None:
    patch_target = "covid_model.explain.shap_calculator.shap.plots.beeswarm"
    with patch(patch_target, wraps=shap.plots.beeswarm) as beeswarm:
        figure = plot_shap_explanation(explanation, max_display=max_display)

    beeswarm.assert_called_once_with(explanation, show=False, max_display=max_display)
    assert figure.number not in existing_figures
    assert plt.fignum_exists(figure.number)
    assert figure.axes
