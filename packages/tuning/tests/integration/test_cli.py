import sys

import pytest
from typer.testing import CliRunner

from covid_tuning.cli import app


@pytest.mark.parametrize("command", ["tune", "eval-tune"])
def test_tuning_subcommands_pass_overrides_and_config_flag_to_hydra(
    command: str,
) -> None:
    arguments = [command, "dataset=interim", "--cfg=job"]
    original_argv = sys.argv

    result = CliRunner().invoke(app, arguments, prog_name="covid-tuning")

    assert result.exit_code == 0, result.output
    assert "path: data/interim/train.csv" in result.output
    assert sys.argv is original_argv
