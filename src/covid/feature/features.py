from pandas import DataFrame

ID = "id"
TARGET = "Long_COVID"


def get_loci_columns(X: DataFrame) -> list[str]:
    return [col for col in X.columns if col.startswith("chr")]
