import pandas as pd

ID = "id"
TARGET = "Long_COVID"


def get_loci_data(data: pd.DataFrame) -> pd.DataFrame:
    return data[get_loci_columns(data)]


def get_loci_columns(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col.startswith("chr")]
