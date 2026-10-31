from dataclasses import dataclass

from covid.tune import TuningSpec


@dataclass(frozen=True)
class TuningEvaluationSpec:
    """Configure inner hyperparameter tuning and repeated outer evaluation folds."""

    inner: TuningSpec
    n_outer_splits: int
    n_outer_repeats: int
