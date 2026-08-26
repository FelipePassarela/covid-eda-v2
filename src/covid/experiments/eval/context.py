from dataclasses import dataclass

import pandas as pd
from imblearn.pipeline import Pipeline


@dataclass(frozen=True)
class EvaluationContext:
    X: pd.DataFrame
    y: pd.Series
    pipeline: Pipeline
