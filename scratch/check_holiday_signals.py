import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def check_special_dates():
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Daily Aggregation
    daily = sales.groupby('Date')['Revenue'].sum().reset_index()
    daily['month'] = daily['Date'].dt.month
    daily['day'] = daily['Date'].dt.day
    daily['year'] = daily['Date'].dt.year
    
    # 3. Calculate Monthly Median for Context
    monthly_median = daily.groupby(['year', 'month'])['Revenue'].transform('median')
    daily['lift'] = daily['Revenue'] / (monthly_median + 1e-6)
    
    # 4. Filter for 01/01 and 31/12
    jan_1st = daily[(daily['month'] == 1) & (daily['day'] == 1)]
    dec_31st = daily[(daily['month'] == 12) & (daily['day'] == 31)]
    
    print("\n--- Signal Check: Jan 1st (01/01) ---")
    print(jan_1st[['Date', 'Revenue', 'lift']].sort_values('Date'))
    print(f"Median Lift for Jan 1st: {jan_1st['lift'].median():.2f}x")
    
    print("\n--- Signal Check: Dec 31st (31/12) ---")
    print(dec_31st[['Date', 'Revenue', 'lift']].sort_values('Date'))
    print(f"Median Lift for Dec 31st: {dec_31st['lift'].median():.2f}x")

if __name__ == "__main__":
    check_special_dates()
