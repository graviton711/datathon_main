import os
from pathlib import Path

class Config:
    # Project Paths
    PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    SUBMISSION_DIR = PROJECT_ROOT / "submissions"
    LOG_DIR = PROJECT_ROOT / "logs"

    # Files
    SALES_TRAIN_FILE = PROCESSED_DATA_DIR / "sales.parquet"
    
    # Core Baseline Hyperparameters
    LGBM_N_ESTIMATORS = 2000
    LGBM_LR = 0.01
    LGBM_NUM_LEAVES = 127
    
    # Target and Features
    DATE_COL = "Date"
    
    @classmethod
    def initialize_dirs(cls):
        for dir_path in [cls.MODEL_DIR, cls.SUBMISSION_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    Config.initialize_dirs()
    print("Project directories initialized.")
