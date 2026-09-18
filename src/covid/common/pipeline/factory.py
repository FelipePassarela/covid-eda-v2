from hydra.utils import instantiate
from imblearn.pipeline import Pipeline
from omegaconf import DictConfig


def build_pipeline_from_config(config: DictConfig) -> Pipeline:
    preprocessor_steps = instantiate(config.pipeline.preprocessor, _convert_="all")
    classifier = instantiate(config.pipeline.classifier, _convert_="all")
    pipeline = Pipeline(steps=[*preprocessor_steps, ("classifier", classifier)])
    pipeline.set_output(transform="pandas")
    return pipeline
