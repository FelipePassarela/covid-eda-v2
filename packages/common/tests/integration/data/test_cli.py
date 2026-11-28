from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from covid_common.cli import app


def test_common_help_lists_the_split_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"], prog_name="covid-common")

    assert result.exit_code == 0, result.output
    assert "split" in result.output


def test_split_subcommand_uses_the_given_paths_and_test_size(
    data_path: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "split output"
    arguments = [
        "split",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--test-size",
        "0.3",
    ]

    result = CliRunner().invoke(app, arguments, prog_name="covid-common")

    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output_dir / "train.csv")) == 70
    assert len(pd.read_csv(output_dir / "test.csv")) == 30
