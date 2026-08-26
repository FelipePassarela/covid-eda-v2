from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExplainingSpec:
    pipeline_path: Path
    train_path: Path
    test_path: Path
    max_display: int = 25
