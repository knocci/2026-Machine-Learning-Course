"""Central project constants.

Edit paths and smoke/full experiment settings here instead of scattering them
through the training code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"

UCI_POWER_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip"
)
WEATHER_URL = "https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles/"

UCI_ZIP_PATH = RAW_DATA_DIR / "household_power_consumption.zip"
UCI_RAW_TXT_PATH = RAW_DATA_DIR / "household_power_consumption.txt"
WEATHER_RAW_CSV_PATH = RAW_DATA_DIR / "weather_monthly.csv"
DAILY_POWER_CSV_PATH = PROCESSED_DATA_DIR / "daily_power.csv"

LOCAL_TRAIN_CANDIDATES = (
    PROJECT_ROOT / "train.csv",
    DATA_DIR / "train.csv",
    PROCESSED_DATA_DIR / "train.csv",
)
LOCAL_TEST_CANDIDATES = (
    PROJECT_ROOT / "test.csv",
    PROJECT_ROOT / "tes.csv",
    DATA_DIR / "test.csv",
    DATA_DIR / "tes.csv",
    PROCESSED_DATA_DIR / "test.csv",
    PROCESSED_DATA_DIR / "tes.csv",
)

INPUT_DAYS = 90
SMOKE_OUTPUT_DAYS = 90
LONG_OUTPUT_DAYS = 365

DEFAULT_DEVICE = "auto"
DEFAULT_SEED = 42
SMOKE_MAX_WINDOWS = 32
SMOKE_EPOCHS = 1
SMOKE_BATCH_SIZE = 8

FULL_BATCH_SIZE = 16
FULL_EPOCHS = 30
FULL_SEEDS = (11, 22, 33, 44, 55)

FLOW_EULER_STEPS = 8
FLOW_NUM_SAMPLES = 4
FLOW_PRIOR_LOSS_WEIGHT = 0.1
FLOW_NOISE_MIN = 0.05
FLOW_NOISE_MAX = 0.5

TARGET_COLUMN = "global_active_power"
WEATHER_COLUMNS = ("RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU")
CALENDAR_COLUMNS = ("month", "day_of_week", "is_weekend", "day_sin", "day_cos")

FEATURE_COLUMNS = (
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
    "month",
    "day_of_week",
    "is_weekend",
    "day_sin",
    "day_cos",
)

METEO_DEPT_STATIONS = {
    "92": "92007001",  # BAGNEUX, close to Sceaux
}
METEO_FALLBACK_STATION = "92007001"
METEO_BASE_URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net"
METEO_PATH = "/data/synchro_ftp/BASE/MENS/"


def ensure_project_dirs() -> None:
    """Create all project output directories."""
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        FIGURE_DIR,
        MODEL_DIR,
        METRIC_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
