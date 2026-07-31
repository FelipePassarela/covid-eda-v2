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

    def as_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_searches": self.n_searches,
            "n_fold_repeats": self.n_fold_repeats,
            "scoring": self.scoring,
            "pipeline": repr(self.pipeline),
            "param_distributions": {
                param: repr(distr) for param, distr in self.param_distributions.items()
            },
        }
