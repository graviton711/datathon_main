import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def analyze_cogs_bimodal():
    print("--- COGS Ratio Bimodal Analysis (August Even/Odd Years) ---")
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    sales['is_odd_year'] = sales['year'] % 2 != 0
    
    # Calculate COGS Ratio
    sales['cogs_ratio'] = sales['COGS'] / (sales['Revenue'] + 1e-6)
    
    # Filter for August (Month 8)
    august = sales[sales['month'] == 8].copy()
    
    # Group by Year Type
    stats = august.groupby('is_odd_year')['cogs_ratio'].agg(['mean', 'median', 'std', 'count'])
    print("\nAugust COGS Ratio Stats:")
    print(stats)
    
    # Detail by year to see the trend
    yearly_stats = august.groupby('year')['cogs_ratio'].median()
    print("\nAugust Median COGS Ratio by Year:")
    print(yearly_stats)
    
    # Global Quantiles (to check if clipping is the issue)
    sales_clean = sales[sales['Revenue'] > 0]
    q01 = sales_clean['cogs_ratio'].quantile(0.01)
    q99 = sales_clean['cogs_ratio'].quantile(0.99)
    print(f"\nGlobal COGS Ratio Clip Range: [{q01:.4f}, {q99:.4f}]")

if __name__ == "__main__":
    analyze_cogs_bimodal()
