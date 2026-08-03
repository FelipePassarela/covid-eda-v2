from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.model_selection import TunedThresholdClassifierCV


class TrainingTracker(Protocol):
    def track_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...

    def track_threshold_tuning(
        self, model: TunedThresholdClassifierCV, scoring: str
    ) -> None: ...

    def track_model(self, model_path: Path) -> None: ...
