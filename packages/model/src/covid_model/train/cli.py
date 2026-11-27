import hydra
from omegaconf import DictConfig

from covid_common import paths, prepare_config_for_wandb
from covid_common.logging import configure_logging, log_config
from covid_model.train import fit, tune_threshold
from covid_model.train.spec_factory import training_spec_from_config
from covid_model.train.wandb_tracker import WAndBTrainingTracker


@hydra.main(
    version_base=None, config_path="pkg://covid_model.conf", config_name="train"
)
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "train.log")
    train(config)


def train(config: DictConfig) -> None:
    log_config(config)

    config_for_wandb = prepare_config_for_wandb(config)
    wandb_tracker = WAndBTrainingTracker(config=config_for_wandb)

    with wandb_tracker as tracker:
        spec = training_spec_from_config(config, tracker)
        if should_tune_threshold(config):
            tune_threshold(spec, scoring=config.tuning_scoring)
        else:
            fit(spec)


def should_tune_threshold(config: DictConfig) -> bool:
    return config.tune_threshold
