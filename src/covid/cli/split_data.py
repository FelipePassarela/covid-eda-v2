from pathlib import Path

import typer
from sklearn.model_selection import train_test_split

from covid import paths
from covid.experiments.shared import constants
from covid.experiments.shared.data import load_data, save_data
from covid.experiments.shared.feature import TARGET


def main() -> None:
    typer.run(split_data)


def split_data(
    data_path: Path = paths.RAW_DATA_PATH,
    output_dir: Path = paths.INTERIM_DATA_DIR,
    test_size: float = 0.2,
    random_state: int | None = constants.RANDOM_STATE,
) -> None:
    data = load_data(data_path)

    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[TARGET]
    )

    save_data(train, output_dir / "train.csv")
    save_data(test, output_dir / "test.csv")


if __name__ == "__main__":
    main()
