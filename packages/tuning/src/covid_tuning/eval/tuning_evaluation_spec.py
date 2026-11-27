from dataclasses import dataclass

from sklearn.model_selection import RepeatedStratifiedKFold

from covid_tuning import TuningSpec


@dataclass(frozen=True)
class TuningEvaluationSpec:
    """Configure inner hyperparameter tuning and repeated outer evaluation folds."""

    inner: TuningSpec
    outer_cv: RepeatedStratifiedKFold
    n_outer_jobs: int = -1
