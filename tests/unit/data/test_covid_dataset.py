from covid.data import CovidDataset


def test_target_not_in_X(dataset: CovidDataset):
    assert "target" not in dataset.X.columns
    assert "target" not in dataset.split()[0].columns


def test_target_is_y(dataset: CovidDataset):
    assert "target" == dataset.y.name
    assert "target" == dataset.split()[1].name
    assert dataset.y.equals(dataset.split()[1])


def test_id_column_removed(dataset: CovidDataset):
    assert "id" not in dataset.to_dataframe().columns


def test_sparse_columns_removed(sparse_dataset: CovidDataset):
    assert "sparse_feature" not in sparse_dataset.X.columns
    assert "sparse_feature" not in sparse_dataset.split()[0].columns
    assert "sparse_feature" not in sparse_dataset.to_dataframe().columns
