from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "longa_nao_vacinados.csv"
INTERIM_TRAIN_DATA_PATH = INTERIM_DATA_DIR / "train.csv"
INTERIM_TEST_DATA_PATH = INTERIM_DATA_DIR / "test.csv"

LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42
