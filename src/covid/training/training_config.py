from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float
    random_state: int | None
    n_folds: int
    shuffle: bool
    metrics: list[str]
