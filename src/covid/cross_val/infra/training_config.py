from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    # TODO: split this into CVConfig and DataSplitConfig
    test_size: float
    random_state: int | None
    n_folds: int
    shuffle: bool
    metrics: list[str]
