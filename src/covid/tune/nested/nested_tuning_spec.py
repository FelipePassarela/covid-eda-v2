from dataclasses import dataclass

from covid.tune import TuningSpec


@dataclass(frozen=True)
class NestedTuningSpec:
    inner: TuningSpec
    n_outer_splits: int
