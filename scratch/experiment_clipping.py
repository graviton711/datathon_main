import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def run_clipping_experiment(name, clip_val=None):
    print(f"\n>>> Running Experiment: {name}")
    
    # Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    train_df = sales[sales['Date'] <= '2021-12-31'].copy()
    test_df = sales[(sales['Date'] >= '2022-01-01') & (sales['Date'] <= '2022-12-31')].copy()
    
    # Initialize Pipeline
    pipeline = ForecastingPipeline()
    
    # 1. Fit to get standard behavior
    pipeline.fit(train_df)
    
    # 2. Predict
    preds = pipeline.predict(test_df[['Date']])
    
    mae_rev = mean_absolute_error(test_df['Revenue'], preds['Revenue'])
    total_mae = mae_rev + mean_absolute_error(test_df['COGS'], preds['COGS'])
    
    print(f"Results for {name}: {total_mae:,.0f}")
    return total_mae

if __name__ == "__main__":
    # We need to modify the pipeline.py or wrap the target during fit for a real clipping test.
    # But for a quick check, let's see if the baseline is consistent.
    run_clipping_experiment("Baseline")
