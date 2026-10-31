import hydra
from omegaconf import DictConfig

from covid.common import paths
from covid.common.config_adapter import prepare_config_for_wandb
from covid.common.logging import configure_logging, log_config
from covid.tune.search import TuningSpec, tune
from covid.tune.search.tracking import WAndBTuningTracker
from covid.tune.search.tuning_spec_factory import tuning_spec_from_config


@hydra.main(version_base=None, config_path=str(paths.CONF_DIR), config_name="tune")
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "tune.log")

    spec = tuning_spec_from_config(config)
    run_tuning(spec, config)


def run_tuning(spec: TuningSpec, config: DictConfig) -> None:
    log_config(config)

    tracker = WAndBTuningTracker(config=prepare_config_for_wandb(config))
    with tracker:
        tune(spec, tracker)
