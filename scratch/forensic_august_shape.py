import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def analyze_august_shape():
    print("--- August Odd Year COGS Ratio Shape Analysis ---")
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['cogs_ratio'] = sales['COGS'] / (sales['Revenue'] + 1e-6)
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    sales['day'] = sales['Date'].dt.day
    
    # Filter for August in Odd Years
    aug_odd = sales[(sales['month'] == 8) & (sales['year'] % 2 == 1)]
    
    # Check by day
    shape = aug_odd.groupby('day')['cogs_ratio'].mean().round(4)
    print("\nDaily Average COGS Ratio in August (Odd Years):")
    print(shape)
    
    # Also check July 30-31 as it's the start of Urban Blowout
    july_start = sales[(sales['month'] == 7) & (sales['day'] >= 30) & (sales['year'] % 2 == 1)]
    print("\nJuly 30-31 Average COGS Ratio (Odd Years):")
    print(july_start.groupby('day')['cogs_ratio'].mean().round(4))

if __name__ == "__main__":
    analyze_august_shape()
