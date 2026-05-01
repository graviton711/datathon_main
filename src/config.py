import os
from pathlib import Path

class Config:
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    SUBMISSION_DIR = PROJECT_ROOT / "submissions"
    LOG_DIR = PROJECT_ROOT / "logs"

    # Files
    SALES_TRAIN_FILE = PROCESSED_DATA_DIR / "sales.parquet"
    WEB_TRAFFIC_FILE = PROCESSED_DATA_DIR / "web_traffic.parquet"
    ORDERS_FILE = PROCESSED_DATA_DIR / "orders.parquet"
    
    # Core Baseline Hyperparameters
    LGBM_N_ESTIMATORS = 2000
    LGBM_LR = 0.01
    LGBM_NUM_LEAVES = 63
    
    # Target and Features
    DATE_COL = "Date"
    
    # Data Weighting
    DECAY_HALF_LIFE_YEARS = 1.5
    REC_HISTORY_WINDOW = 60
    REC_LAG_WINDOW = 30
    
    # Momentum & Scaling
    MOMENTUM_WINDOWS = [90, 180, 270, 365]
    MOMENTUM_DECAY_DAYS = 180
    MOMENTUM_CLIP_MIN = 0.5
    MOMENTUM_CLIP_MAX = 2.0
    
    # Discovery & Cleaning
    EVENT_LIFT_THRESHOLD = 1.10
    EVENT_MIN_OCCURRENCES = 5
    TET_WINDOW_DAYS = 2
    TET_CONTAMINATION_DAYS = 15
    
    # Forecast Damping (Market Correction)
    DAMPING_Y1 = 0.85
    DAMPING_Y2 = 0.3

    # Trailing Momentum Floor (prevents seasonal over-suppression in Sep)
    # Alpha is computed from training data in fit() via MarketAnalyst.calculate_seasonal_floor_alpha()
    SEP_OCT_FLOOR_MONTHS = [9]
    SEP_OCT_FLOOR_WINDOW = 60

    # Oct floor (computed separately — lower historical alpha than Sep)
    OCT_FLOOR_MONTHS = [10]
    
    @classmethod
    def initialize_dirs(cls):
        for dir_path in [cls.MODEL_DIR, cls.SUBMISSION_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    Config.initialize_dirs()
    print("Project directories initialized.")
