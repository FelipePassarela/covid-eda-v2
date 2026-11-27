import hydra
from omegaconf import DictConfig

from covid_common import paths
from covid_common.config_adapter import prepare_config_for_wandb
from covid_common.logging import configure_logging
from covid_tuning.eval.tracking.wandb_tuning_evaluation_tracker import (
    WandBTuningEvaluationTracker,
)
from covid_tuning.eval.tuning_evaluation import evaluate_tuning
from covid_tuning.eval.tuning_evaluation_spec import TuningEvaluationSpec
from covid_tuning.eval.tuning_evaluation_spec_factory import (
    tuning_evaluation_spec_from_config,
)


@hydra.main(
    version_base=None,
    config_path="pkg://covid_tuning.conf",
    config_name="tuning-evaluation",
)
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "tuning-evaluation.log")

    spec = tuning_evaluation_spec_from_config(config)
    run_tuning_evaluation(spec, config)


def run_tuning_evaluation(spec: TuningEvaluationSpec, config: DictConfig) -> None:
    tracker = WandBTuningEvaluationTracker(
        config=prepare_config_for_wandb(config), run_name=config.name
    )
    with tracker:
        evaluate_tuning(spec, tracker)
