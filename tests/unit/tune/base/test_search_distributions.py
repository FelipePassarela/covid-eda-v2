from importlib.resources import files
from importlib.resources.abc import Traversable

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optuna.distributions import BaseDistribution


def _search_distribution_configs() -> list[Traversable]:
    distributions = files("covid_tuning.conf").joinpath("search", "param_distributions")
    configs = [
        config
        for group in ("classifier", "selector")
        for config in distributions.joinpath(group).iterdir()
        if config.name.endswith(".yaml")
    ]
    return sorted(configs, key=lambda config: config.name)


@pytest.mark.parametrize(
    "config", _search_distribution_configs(), ids=lambda config: config.name
)
def test_yaml_instantiates_optuna_distributions(config: Traversable) -> None:
    with config.open("r", encoding="utf-8") as file:
        distributions = instantiate(OmegaConf.load(file), _convert_="all")

    assert distributions
    assert all(isinstance(value, BaseDistribution) for value in distributions.values())
