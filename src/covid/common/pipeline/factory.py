from typing import Any

from hydra.utils import instantiate
from imblearn.pipeline import Pipeline
from omegaconf import DictConfig


def build_pipeline_from_config(config: DictConfig) -> Pipeline:
    preprocessor_steps = _build_preprocessor(config.pipeline.preprocessor)
    classifier = instantiate(config.pipeline.classifier, _convert_="all")
    pipeline = Pipeline(steps=[*preprocessor_steps, ("classifier", classifier)])
    pipeline.set_output(transform="pandas")
    return pipeline


def _build_preprocessor(config: DictConfig) -> list[tuple[str, Any]]:
    names = list(config.order)
    if len(names) != len(set(names)):
        raise ValueError("Steps names must be unique")

    missing = [name for name in names if name not in config.steps]
    if missing:
        raise ValueError(f"Steps not found in steps: {missing}")

    return [(name, instantiate(config.steps[name], _convert_="all")) for name in names]
