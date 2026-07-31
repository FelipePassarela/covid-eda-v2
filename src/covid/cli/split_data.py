from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from covid import constants
from covid.data import load_data
from covid.feature import TARGET


def main() -> None:
    typer.run(split_data)


def split_data(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int | None = constants.RANDOM_STATE,
) -> None:
    data = load_data(data_path)
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[TARGET]
    )
    save_datasets(train, test, output_dir)


def save_datasets(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    train.to_csv(train_path, index=False)
    typer.echo(f"Train set ({len(train)} samples) saved to {train_path}")

    test_path = output_dir / "test.csv"
    test.to_csv(test_path, index=False)
    typer.echo(f"Test set ({len(test)} samples) saved to {test_path}")


if __name__ == "__main__":
    main()
