import os
from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Files
    SALES_TRAIN = RAW_DATA_DIR / "sales.csv"
    SALES_TEST = RAW_DATA_DIR / "sample_submission.csv"
    
    # Model Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Time Series Params
    START_DATE = "2012-07-04"
    TRAIN_END_DATE = "2022-12-31"
    TEST_START_DATE = "2023-01-01"
    TEST_END_DATE = "2024-07-01"

config = Config()
