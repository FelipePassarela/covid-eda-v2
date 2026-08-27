from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.model_selection import TunedThresholdClassifierCV

from covid.common.data import load_and_split_data


def load_pipeline(pipeline_path: Path) -> Pipeline:
    return joblib.load(pipeline_path)


def load_and_transform_features(
    preprocessor: Pipeline, data_path: Path
) -> pd.DataFrame:
    X_train, _ = load_and_split_data(data_path)
    return preprocessor.transform(X_train)


def save_pipeline(
    model: Pipeline | TunedThresholdClassifierCV, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.success("Pipeline saved to {}", output_path)
