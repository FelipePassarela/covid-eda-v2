import hydra
from omegaconf import DictConfig

from covid.common import paths
from covid.common.config_adapter import prepare_config_for_wandb
from covid.common.logging import configure_logging
from covid.tune.base.config_adapter import tuning_spec_from_config
from covid.tune.nested.nested_tune import nested_tune
from covid.tune.nested.nested_tuning_spec import NestedTuningSpec
from covid.tune.nested.tracking.wandb_nested_tracker import WandBNestedTuningTracker


@hydra.main(version_base=None, config_path="../../conf", config_name="nested-tune")
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "nested-tune.log")

    spec = create_spec(config)
    run_nested_tune(spec, config)


def create_spec(config: DictConfig) -> NestedTuningSpec:
    return NestedTuningSpec(
        inner=tuning_spec_from_config(config.inner_search),
        outer_n_splits=config.outer_n_splits,
    )


def run_nested_tune(spec: NestedTuningSpec, config: DictConfig) -> None:
    tracker = WandBNestedTuningTracker(
        config=prepare_config_for_wandb(config), run_name=config.name
    )
    with tracker:
        nested_tune(spec, tracker)
