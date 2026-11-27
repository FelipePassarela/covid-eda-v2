from unittest.mock import patch

from covid_model.train.wandb_tracker import WAndBTrainingTracker


def test_training_tracker_passes_run_group_to_wandb() -> None:
    run_group = "experiments"
    config = {"name": "training", "wandb": {"group": run_group}}

    with patch("wandb.init") as init, WAndBTrainingTracker(config=config):
        init.assert_called_once()
        assert init.call_args.kwargs["group"] == run_group
