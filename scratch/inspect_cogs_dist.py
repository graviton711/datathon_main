import pandas as pd
import numpy as np
import sys
from pathlib import Path
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def inspect_cogs_distribution():
    print("--- Inspecting COGS Prediction Distribution (August 2021) ---")
    raw_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    raw_sales['Date'] = pd.to_datetime(raw_sales['Date'])
    
    # Train up to 2020, Test 2021
    train = raw_sales[raw_sales['Date'].dt.year <= 2020].copy()
    test = raw_sales[raw_sales['Date'].dt.year == 2021].copy()
    
    p = ForecastingPipeline()
    p._validate_feature_contract = lambda x: None
    p.fit(train)
    
    # Add Lags (Absolute values from history)
    all_data = pd.concat([train.tail(14), test])
    all_data = p._add_lags(all_data)
    test_lags = all_data.iloc[14:].copy()
    
    # Get features (which now include is_odd_year_aug)
    X_test = p.revenue_pipeline.named_steps['features'].transform(test_lags)
    
    # Predict raw ratio
    cogs_model = p.cogs_pipeline.named_steps['model']
    raw_preds = cogs_model.predict(X_test)
    
    # Analysis for August
    aug_mask = (test_lags['Date'].dt.month == 8).values
    aug_raw = raw_preds[aug_mask]
    
    print(f"\nAugust 2021 Raw Prediction Stats:")
    print(f"  Min:    {aug_raw.min():.4f}")
    print(f"  Max:    {aug_raw.max():.4f}")
    print(f"  Mean:   {aug_raw.mean():.4f}")
    print(f"  Median: {np.median(aug_raw):.4f}")
    
    print(f"\nCurrent System Clip: {p.cogs_ratio_clip}")
    
    # Check how many are hitting the ceiling
    hitting_ceiling = np.sum(aug_raw >= p.cogs_ratio_clip[1])
    print(f"\nDays hitting the clip ceiling ({p.cogs_ratio_clip[1]:.4f}): {hitting_ceiling} / 31")

if __name__ == "__main__":
    inspect_cogs_distribution()
