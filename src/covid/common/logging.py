from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="5 MB")


def log_config(config: DictConfig) -> None:
    logger.info("Loaded configuration: {}", OmegaConf.to_yaml(config))
