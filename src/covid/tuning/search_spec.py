from dataclasses import dataclass, field
from typing import Any

from imblearn.pipeline import Pipeline


@dataclass
class RandomizedSearchSpec:
    name: str
    pipeline: Pipeline
    param_distributions: dict[str, Any]
    n_searches: int
    n_fold_repeats: int = 5
    scoring: list[str] = field(default_factory=lambda: ["balanced_accuracy"])
