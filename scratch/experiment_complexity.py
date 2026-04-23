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

def run_experiment(name, use_inventory=False, num_leaves=63, clip_val=None):
    print(f"\n>>> Running Experiment: {name}")
    
    # Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    train_df = sales[sales['Date'] <= '2021-12-31'].copy()
    test_df = sales[(sales['Date'] >= '2022-01-01') & (sales['Date'] <= '2022-12-31')].copy()
    
    # 1. Initialize Pipeline
    pipeline = ForecastingPipeline()
    
    # 2. Apply Experiment Settings
    pipeline.revenue_pipeline.named_steps['model'].set_params(num_leaves=num_leaves)
    
    # 3. Fit
    pipeline.fit(train_df)
    
    # 4. Predict
    preds = pipeline.predict(test_df[['Date']])
    
    mae_rev = mean_absolute_error(test_df['Revenue'], preds['Revenue'])
    mae_cogs = mean_absolute_error(test_df['COGS'], preds['COGS'])
    total_mae = mae_rev + mae_cogs
    
    print(f"Results for {name}:")
    print(f"  Revenue MAE: {mae_rev:,.0f}")
    print(f"  COGS MAE:    {mae_cogs:,.0f}")
    print(f"  TOTAL MAE:   {total_mae:,.0f}")
    return total_mae

if __name__ == "__main__":
    results = {}
    results['Baseline'] = run_experiment("Baseline (63 leaves)")
    results['Complexity'] = run_experiment("Complexity (127 leaves)", num_leaves=127)
    
    print("\n=== Experiment Summary ===")
    for k, v in results.items():
        print(f"{k}: {v:,.0f}")
