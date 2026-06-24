from dataclasses import dataclass


@dataclass(frozen=True)
class DataCleaningConfig:
    target_column: str
    id_column: str
    sparse_threshold: float
