from typing import Protocol

import pandas as pd


class CVReport(Protocol):
    def to_dataframe(self) -> pd.DataFrame:
        pass
