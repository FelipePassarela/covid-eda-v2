import pandas as pd
from sklearn.pipeline import Pipeline


def select_best_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selector: Pipeline,
    allow_data_leakage: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects the best features based on their importance scores.

    Args:
        X_train (pd.DataFrame): The input DataFrame containing the training features.
        X_test (pd.DataFrame): The input DataFrame containing the test features.
        y_train (pd.Series): The target values for the training set.
        y_test (pd.Series): The target values for the test set.
        selector (Pipeline): An instance of a pipeline containing the feature selector.
        allow_data_leakage (bool): Whether to allow data leakage during feature selection. Default is False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the selected features for the training and test sets.
    """
    if not allow_data_leakage:
        selector.fit(X_train, y_train)
    else:
        X_full = pd.concat([X_train, X_test], axis=0)
        y_full = pd.concat([y_train, y_test], axis=0)
        selector.fit(X_full, y_full)

    feature_names = selector.get_feature_names_out()
    X_train_selected = pd.DataFrame(
        selector.transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        selector.transform(X_test), columns=feature_names, index=X_test.index
    )

    return X_train_selected, X_test_selected
