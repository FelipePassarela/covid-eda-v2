from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from covid.feature import TARGET


def split_data(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> None:
    typer.echo(f"Reading data from {data_path}")

    data = pd.read_csv(data_path)
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[TARGET]
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    typer.echo(f"Train set ({len(train)} samples) saved to {train_path}")
    typer.echo(f"Test set ({len(test)} samples) saved to {test_path}")


def main():
    typer.run(split_data)


if __name__ == "__main__":
    main()
