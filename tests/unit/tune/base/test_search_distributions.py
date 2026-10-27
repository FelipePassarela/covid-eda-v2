from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optuna.distributions import BaseDistribution

SEARCH_ROOT = (
    Path(__file__).resolve().parents[4] / "src/covid/conf/search/param_distributions"
)


@pytest.mark.parametrize(
    "path", sorted(SEARCH_ROOT.rglob("*.yaml")), ids=lambda p: p.stem
)
def test_yaml_instantiates_optuna_distributions(path: Path) -> None:
    distributions = instantiate(OmegaConf.load(path), _convert_="all")

    assert distributions
    assert all(isinstance(value, BaseDistribution) for value in distributions.values())
