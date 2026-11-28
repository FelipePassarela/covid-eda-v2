import sys

from typer.testing import CliRunner

from covid_model.cli import app


def test_train_subcommand_passes_override_and_config_flag_to_hydra() -> None:
    arguments = ["train", "pipeline=xgboost", "--cfg", "job"]
    original_argv = sys.argv

    result = CliRunner().invoke(app, arguments, prog_name="covid-model")

    assert result.exit_code == 0, result.output
    assert "_target_: xgboost.XGBClassifier" in result.output
    assert sys.argv is original_argv
