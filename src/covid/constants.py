from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

LONG_COVID_RAW_DATA_PATH = RAW_DATA_DIR / "longa_nao_vacinados.csv"
