import pandas as pd
import pytest

from covid_tuning.eval.visualization.data import BoxplotData


def test_boxplot_data_preserves_model_and_fold_order() -> None:
    data = BoxplotData(
        model_names=["svm", "dummy"],
        folds_scores=[[0.8, 0.7], [0.4, 0.5]],
        metric="test_balanced_accuracy",
    )

    expected = pd.DataFrame(
        {
            "model": ["svm", "svm", "dummy", "dummy"],
            "test_balanced_accuracy": [0.8, 0.7, 0.4, 0.5],
        }
    )
    pd.testing.assert_frame_equal(data.to_long_dataframe(), expected)
    assert data.metric == "test_balanced_accuracy"
    assert data.n_folds == 2


@pytest.mark.parametrize(
    ("model_names", "folds_scores", "message"),
    [
        pytest.param([], [], "At least one model", id="no-models"),
        pytest.param(["svm"], [], "must have the same length", id="missing-scores"),
        pytest.param(
            ["svm", "dummy"],
            [[0.8, 0.7], [0.4]],
            "inconsistent fold counts",
            id="unequal-fold-counts",
        ),
    ],
)
def test_boxplot_data_rejects_invalid_inputs(
    model_names: list[str], folds_scores: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BoxplotData(model_names, folds_scores, metric="test_balanced_accuracy")
