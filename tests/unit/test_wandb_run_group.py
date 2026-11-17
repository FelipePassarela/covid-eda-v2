from typing import Any
from unittest.mock import Mock

import pytest

import wandb
from covid.train.wandb_tracker import WAndBTrainingTracker
from covid.tune.eval.tracking.wandb_tuning_evaluation_tracker import (
    WandBTuningEvaluationTracker,
)
from covid.tune.search.tracking.wandb_tuning_tracker import WAndBTuningTracker


@pytest.mark.parametrize(
    "tracker_type",
    [WAndBTrainingTracker, WAndBTuningTracker, WandBTuningEvaluationTracker],
)
@pytest.mark.parametrize(
    ("tracking_config", "expected_group"),
    [
        pytest.param({"wandb": {"group": "experiments"}}, "experiments", id="group"),
        pytest.param({}, None, id="no-wandb-section"),
        pytest.param({"wandb": {}}, None, id="no-group"),
        pytest.param({"wandb": {"group": None}}, None, id="null-group"),
    ],
)
def test_tracker_passes_run_group_to_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tracker_type: type[
        WAndBTrainingTracker | WAndBTuningTracker | WandBTuningEvaluationTracker
    ],
    tracking_config: dict[str, Any],
    expected_group: str | None,
) -> None:
    init = Mock()
    monkeypatch.setattr(wandb, "init", init)

    with tracker_type(config={"name": "test-run", **tracking_config}):
        init.assert_called_once()
        assert init.call_args.kwargs["group"] == expected_group
