import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import imread

from covid.tune.eval.result.visualization.cli import main
from covid.tune.eval.result.visualization.io import create_boxplot_data


def test_create_boxplot_data_reads_selected_metric_from_each_csv(
    cv_results_paths: list[Path],
) -> None:
    data = create_boxplot_data(cv_results_paths, metric="test_roc_auc")

    expected = pd.DataFrame(
        {
            "model": ["svm", "svm", "svm", "dummy", "dummy", "dummy"],
            "test_roc_auc": [0.90, 0.85, 0.95, 0.60, 0.65, 0.55],
        }
    )
    pd.testing.assert_frame_equal(data.to_long_dataframe(), expected)


def test_create_boxplot_data_rejects_missing_metric(
    cv_results_paths: list[Path],
) -> None:
    with pytest.raises(KeyError, match="test_f1"):
        create_boxplot_data(cv_results_paths, metric="test_f1")


@pytest.mark.parametrize(
    ("metric_args", "metric_label"),
    [
        pytest.param([], "Balanced Accuracy", id="default-metric"),
        pytest.param(["--metric", "test_roc_auc"], "ROC AUC", id="selected-metric"),
    ],
)
def test_boxplot_command_saves_labeled_png(
    cv_results_paths: list[Path],
    metric_args: list[str],
    metric_label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_figures: set[int],
) -> None:
    output_path = tmp_path / "reports" / "boxplot.png"
    _set_boxplot_argv(monkeypatch, cv_results_paths, output_path, metric_args)

    with (
        plt.rc_context(),
        patch.object(plt, "show"),
        pytest.raises(SystemExit) as exit_info,
    ):
        main()

    assert exit_info.value.code == 0

    axes = _new_axes(existing_figures)
    assert axes.has_data()
    assert axes.get_xlabel() == metric_label
    assert axes.get_ylabel() == "Model"
    assert [tick.get_text() for tick in axes.get_yticklabels()] == ["SVM", "Dummy"]
    assert "3 Outer Folds" in axes.get_title()

    assert imread(output_path).size > 0


@pytest.fixture()
def cv_results_paths(tmp_path: Path) -> list[Path]:
    svm_path = tmp_path / "svm.csv"
    dummy_path = tmp_path / "dummy.csv"

    pd.DataFrame(
        {
            "test_balanced_accuracy": [0.75, 0.70, 0.80],
            "test_roc_auc": [0.90, 0.85, 0.95],
        }
    ).to_csv(svm_path, index=False)
    pd.DataFrame(
        {
            "test_balanced_accuracy": [0.50, 0.55, 0.45],
            "test_roc_auc": [0.60, 0.65, 0.55],
        }
    ).to_csv(dummy_path, index=False)

    return [svm_path, dummy_path]


def _set_boxplot_argv(
    monkeypatch: pytest.MonkeyPatch,
    cv_results_paths: list[Path],
    output_path: Path,
    metric_args: list[str],
) -> None:
    argv = [
        "boxplot",
        *(str(path) for path in cv_results_paths),
        "--output-path",
        str(output_path),
        *metric_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)


def _new_axes(existing_figures: set[int]) -> Axes:
    (figure_number,) = set(plt.get_fignums()) - existing_figures
    (axes,) = plt.figure(figure_number).axes
    return axes
