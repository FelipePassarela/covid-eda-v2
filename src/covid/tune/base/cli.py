import hydra
from omegaconf import DictConfig

from covid.common import paths
from covid.common.config_adapter import prepare_config_for_wandb
from covid.common.logging import configure_logging, log_config
from covid.tune.base import TuningSpec, tune
from covid.tune.base.config_adapter import tuning_spec_from_config
from covid.tune.base.tracking import WAndBTuningTracker


@hydra.main(version_base=None, config_path=str(paths.CONF_DIR), config_name="tune")
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "tune.log")

    spec = tuning_spec_from_config(config)
    run_tuning(spec, config)


def run_tuning(spec: TuningSpec, config: DictConfig) -> None:
    log_config(config)

    tracker = WAndBTuningTracker(
        config=prepare_config_for_wandb(config), run_name=spec.name
    )
    with tracker:
        tune(spec, tracker)
