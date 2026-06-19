from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    raw_path: Path
    target_column: str
    id_column: str
    sparse_threshold: float
