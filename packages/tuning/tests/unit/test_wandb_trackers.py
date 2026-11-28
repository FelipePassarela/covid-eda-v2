from unittest.mock import patch

import pytest

from covid_tuning.eval.tracking import WandBTuningEvaluationTracker
from covid_tuning.search.tracking import WAndBTuningTracker


@pytest.mark.parametrize(
    "tracker_type",
    [WAndBTuningTracker, WandBTuningEvaluationTracker],
)
def test_tuning_trackers_pass_run_group_to_wandb(
    tracker_type: type[WAndBTuningTracker | WandBTuningEvaluationTracker],
) -> None:
    run_group = "experiments"
    config = {"name": "test-run", "wandb": {"group": run_group}}

    with patch("wandb.init") as init, tracker_type(config=config):
        init.assert_called_once()
        assert init.call_args.kwargs["group"] == run_group
