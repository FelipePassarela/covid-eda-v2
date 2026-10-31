from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline

from covid.eval.context import EvaluationContext


@pytest.fixture()
def prediction_pipeline() -> MagicMock:
    pipeline = MagicMock(spec=Pipeline)
    pipeline.predict.return_value = np.array([0, 1, 0, 1])
    pipeline.predict_proba.return_value = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )
    return pipeline


@pytest.fixture()
def evaluation_context(prediction_pipeline: MagicMock) -> EvaluationContext:
    index = pd.Index([42, 7, 99, 12], name="sample")
    features = pd.DataFrame({"measurement": [1.0, 2.0, 3.0, 4.0]}, index=index)
    target = pd.Series([0, 0, 1, 1], index=index, name="target")
    return EvaluationContext(features, target, prediction_pipeline)
