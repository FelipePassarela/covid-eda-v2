import pytest

from covid_common.config_adapter import get_wandb_run_group


@pytest.mark.parametrize(
    ("config", "expected_group"),
    [
        pytest.param({"wandb": {"group": "experiments"}}, "experiments", id="group"),
        pytest.param({}, None, id="no-wandb-section"),
        pytest.param({"wandb": {}}, None, id="no-group"),
        pytest.param({"wandb": {"group": None}}, None, id="null-group"),
    ],
)
def test_get_wandb_run_group_reads_optional_group(
    config: dict[str, dict[str, str | None]], expected_group: str | None
) -> None:
    assert get_wandb_run_group(config) == expected_group
