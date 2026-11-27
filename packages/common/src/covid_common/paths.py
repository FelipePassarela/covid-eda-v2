import tomllib
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for directory in (start, *start.parents):
        pyproject = directory / "pyproject.toml"
        lockfile = directory / "uv.lock"
        if not lockfile.is_file() or not pyproject.is_file():
            continue
        with pyproject.open("rb") as file:
            project = tomllib.load(file).get("project", {})
        if project.get("name") == "covid-eda-v2":
            return directory
    return start


ROOT_DIR = _find_project_root(Path.cwd())

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "longa_nao_vacinados.csv"
INTERIM_TRAIN_DATA_PATH = INTERIM_DATA_DIR / "train.csv"
INTERIM_TEST_DATA_PATH = INTERIM_DATA_DIR / "test.csv"

LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
