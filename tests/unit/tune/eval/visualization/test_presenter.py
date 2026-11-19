import pandas as pd
import pytest

from covid.tune.eval.result.visualization.data import BoxplotData
from covid.tune.eval.result.visualization.presenter import (
    format_boxplot_dataframe,
    present_metric,
)


def test_format_boxplot_dataframe_presents_model_names_and_preserves_scores() -> None:
    data = BoxplotData(
        model_names=["logistic-regression", "custom-model"],
        folds_scores=[[0.8, 0.9], [0.6, 0.7]],
        metric="test_roc_auc",
    )

    expected = pd.DataFrame(
        {
            "model": [
                "Logistic Regression",
                "Logistic Regression",
                "custom-model",
                "custom-model",
            ],
            "test_roc_auc": [0.8, 0.9, 0.6, 0.7],
        }
    )
    pd.testing.assert_frame_equal(format_boxplot_dataframe(data), expected)


@pytest.mark.parametrize(
    ("metric", "label"),
    [
        ("test_balanced_accuracy", "Balanced Accuracy"),
        ("train_roc_auc", "ROC AUC (Train)"),
        ("test_custom_score", "test_custom_score"),
    ],
)
def test_present_metric_uses_display_label_when_available(
    metric: str, label: str
) -> None:
    assert present_metric(metric) == label
