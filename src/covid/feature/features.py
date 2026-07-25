from pandas import DataFrame

ID = "id"
TARGET = "Long_COVID"


def get_loci_data(data: DataFrame) -> DataFrame:
    return data[get_loci_columns(data)]


def get_loci_columns(data: DataFrame) -> list[str]:
    return [col for col in data.columns if col.startswith("chr")]
