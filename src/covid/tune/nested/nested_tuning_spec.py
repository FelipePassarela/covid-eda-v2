from dataclasses import dataclass

from covid.tune import TuningSpec


@dataclass(frozen=True)
class NestedTuningSpec:
    inner: TuningSpec
    outer_n_splits: int
