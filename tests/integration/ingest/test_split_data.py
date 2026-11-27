from pathlib import Path

import pandas as pd
import pytest

from covid_common import constants
from covid_common.feature import ID, TARGET
from covid_common.ingest.split_data import split_data


def read_split(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(output_dir / "train.csv", dtype={ID: str}),
        pd.read_csv(output_dir / "test.csv", dtype={ID: str}),
    )


@pytest.mark.parametrize(("test_size", "expected_test_rows"), [(0.2, 20), (0.3, 30)])
def test_split_writes_stratified_partitions_without_losing_or_duplicating_rows(
    data_path: Path,
    source_data: pd.DataFrame,
    tmp_path: Path,
    test_size: float,
    expected_test_rows: int,
) -> None:
    output_dir = tmp_path / "new" / "splits"

    split_data(data_path, output_dir, test_size=test_size)

    train, test = read_split(output_dir)
    assert len(train) == len(source_data) - expected_test_rows
    assert len(test) == expected_test_rows
    assert train[TARGET].mean() == pytest.approx(0.2)
    assert test[TARGET].mean() == pytest.approx(0.2)
    assert set(train[ID]).isdisjoint(test[ID])
    assert list(train.columns) == list(source_data.columns)
    assert list(test.columns) == list(source_data.columns)

    combined = pd.concat([train, test]).sort_values(ID).reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, source_data)


def test_split_default_seed_matches_explicit_seed(
    data_path: Path, tmp_path: Path
) -> None:
    default_dir = tmp_path / "default"
    explicit_dir = tmp_path / "explicit"

    split_data(data_path, default_dir)
    split_data(data_path, explicit_dir, random_state=constants.RANDOM_STATE)

    default_train, default_test = read_split(default_dir)
    explicit_train, explicit_test = read_split(explicit_dir)
    pd.testing.assert_frame_equal(default_train, explicit_train)
    pd.testing.assert_frame_equal(default_test, explicit_test)


def test_split_is_reproducible_with_a_custom_seed(
    data_path: Path, tmp_path: Path
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    split_data(data_path, first_dir, random_state=17)
    split_data(data_path, second_dir, random_state=17)

    first_train, first_test = read_split(first_dir)
    second_train, second_test = read_split(second_dir)
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_changes_partition_with_a_different_seed(
    data_path: Path, tmp_path: Path
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    split_data(data_path, first_dir, random_state=17)
    split_data(data_path, second_dir, random_state=42)

    first_train, first_test = read_split(first_dir)
    second_train, second_test = read_split(second_dir)
    assert set(first_train[ID]) != set(second_train[ID])
    assert set(first_test[ID]) != set(second_test[ID])


@pytest.mark.parametrize("test_size", [-0.1, 0.0, 1.0, 1.1])
def test_split_rejects_invalid_test_sizes_without_writing_files(
    data_path: Path, tmp_path: Path, test_size: float
) -> None:
    output_dir = tmp_path / "splits"

    with pytest.raises(ValueError):
        split_data(data_path, output_dir, test_size=test_size)

    assert not output_dir.exists()


def test_split_rejects_data_without_target(
    data_path: Path, source_data: pd.DataFrame, tmp_path: Path
) -> None:
    source_data.drop(columns=TARGET).to_csv(data_path, index=False)
    output_dir = tmp_path / "splits"

    with pytest.raises(KeyError, match=TARGET):
        split_data(data_path, output_dir)

    assert not output_dir.exists()


def test_split_rejects_a_class_with_only_one_sample(
    data_path: Path, source_data: pd.DataFrame, tmp_path: Path
) -> None:
    source_data[TARGET] = [0] * 99 + [1]
    source_data.to_csv(data_path, index=False)
    output_dir = tmp_path / "splits"

    with pytest.raises(ValueError, match="least populated class"):
        split_data(data_path, output_dir)

    assert not output_dir.exists()


def test_split_propagates_missing_file_errors(tmp_path: Path) -> None:
    output_dir = tmp_path / "splits"

    with pytest.raises(FileNotFoundError):
        split_data(tmp_path / "missing.csv", output_dir)

    assert not output_dir.exists()
