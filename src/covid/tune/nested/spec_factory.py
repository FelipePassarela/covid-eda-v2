from omegaconf import DictConfig

from covid.tune.base.spec_factory import tuning_spec_from_config
from covid.tune.nested import NestedTuningSpec


def nested_tuning_spec_from_config(config: DictConfig) -> NestedTuningSpec:
    return NestedTuningSpec(
        inner=tuning_spec_from_config(config.inner_search),
        n_outer_splits=config.n_outer_splits,
    )
