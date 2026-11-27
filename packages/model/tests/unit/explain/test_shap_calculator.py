import pandas as pd
import pytest
from shap import Explanation

from covid_model.explain.shap_calculator import (
    calculate_shap_importances,
    extract_background_from,
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
