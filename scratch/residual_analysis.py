import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def analyze_target_and_residuals():
    print("--- Analyzing Target Distribution and Residuals ---")
    
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Replicate the Normalization logic from Pipeline
    sales['year'] = sales['Date'].dt.year
    annual_scales = sales.groupby('year')['Revenue'].median().to_dict()
    sales['Revenue_norm'] = sales['Revenue'] / sales['year'].map(annual_scales).replace(0, 1.0)
    
    # 3. Check Distribution Stats
    stats = sales['Revenue_norm'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print("\nNormalized Revenue Stats:")
    print(stats)
    
    # 4. Identify Potential Outliers (e.g. > 3 std or > Q99)
    q99 = stats['99%']
    outliers = sales[sales['Revenue_norm'] > q99]
    print(f"\nFound {len(outliers)} rows above Q99 ({q99:.2f})")
    print(outliers[['Date', 'Revenue', 'Revenue_norm']].head(10))
    
    # 5. Quick Check on Units Received correlation with Residuals
    # (We need a trained model for this)
    print("\nFitting a quick model to check residuals...")
    train_df = sales[sales['year'] <= 2021].copy()
    test_df = sales[sales['year'] == 2022].copy()
    
    pipeline = ForecastingPipeline()
    pipeline.fit(train_df)
    preds = pipeline.predict(test_df[['Date']])
    
    test_df['p_Revenue'] = preds['Revenue'].values
    test_df['residual'] = test_df['Revenue'] - test_df['p_Revenue']
    test_df['abs_error'] = test_df['residual'].abs()
    
    # Load inventory for units_received check
    try:
        inventory = pd.read_csv(Config.RAW_DATA_DIR / "inventory.csv")
        inventory['snapshot_date'] = pd.to_datetime(inventory['snapshot_date'])
        # Aggregate to monthly
        monthly_units = inventory.groupby(['snapshot_date'])['units_received'].sum().reset_index()
        monthly_units['year'] = monthly_units['snapshot_date'].dt.year
        monthly_units['month'] = monthly_units['snapshot_date'].dt.month
        
        test_df['year'] = test_df['Date'].dt.year
        test_df['month'] = test_df['Date'].dt.month
        test_df = pd.merge(test_df, monthly_units[['year', 'month', 'units_received']], on=['year', 'month'], how='left')
        
        corr = test_df[['abs_error', 'units_received']].corr().iloc[0, 1]
        print(f"\nCorrelation between Absolute Error and Monthly Units Received: {corr:.3f}")
    except Exception as e:
        print(f"\nCould not analyze units_received: {e}")

    # Top 10 errors
    print("\nTop 10 Dates with Highest Absolute Error:")
    print(test_df.sort_values('abs_error', ascending=False)[['Date', 'Revenue', 'p_Revenue', 'abs_error']].head(10))

if __name__ == "__main__":
    analyze_target_and_residuals()
