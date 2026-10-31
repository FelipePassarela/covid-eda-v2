from unittest.mock import patch

import pandas as pd
import pytest
import shap
from matplotlib import pyplot as plt
from shap import Explanation

from covid.explain.shap_calculator import (
    calculate_shap_importances,
    extract_background_from,
    plot_shap_explanation,
)


def test_importances_use_absolute_values_and_sort_descending(
    explanation: Explanation,
) -> None:
    importances = calculate_shap_importances(explanation)

    expected = pd.DataFrame(
        {
            "feature": ["large", "small", "tiny"],
            "mean_abs_shap": [3.0, 2.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(importances, expected)


@pytest.mark.parametrize("n_samples", [None, 3, 20])
def test_background_preserves_rows_and_samples_reproducibly(
    n_samples: int | None,
) -> None:
    foreground = pd.DataFrame(
        {"measurement": range(8), "other": range(10, 18)},
        index=pd.Index(range(100, 108), name="sample"),
    )

    background = extract_background_from(foreground, n_samples=n_samples)

    expected_size = (
        len(foreground) if n_samples is None else min(n_samples, len(foreground))
    )
    assert len(background) == expected_size
    assert background.index.is_unique
    pd.testing.assert_frame_equal(background, foreground.loc[background.index])
    pd.testing.assert_frame_equal(
        background, extract_background_from(foreground, n_samples=n_samples)
    )
    if n_samples is None:
        pd.testing.assert_frame_equal(background, foreground)


@pytest.mark.parametrize("max_display", [2, 25])
def test_plot_creates_figure_without_displaying_it(
    explanation: Explanation, max_display: int, existing_figures: set[int]
) -> None:
    patch_target = "covid.explain.shap_calculator.shap.plots.beeswarm"
    with patch(patch_target, wraps=shap.plots.beeswarm) as beeswarm:
        figure = plot_shap_explanation(explanation, max_display=max_display)

    beeswarm.assert_called_once_with(explanation, show=False, max_display=max_display)
    assert figure.number not in existing_figures
    assert plt.fignum_exists(figure.number)
    assert figure.axes
