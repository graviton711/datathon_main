import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data")

def analyze_procurement():
    print("Loading inventory and sales data...")
    # Try loading processed first, fallback to raw
    try:
        inv = pd.read_parquet(DATA_DIR / "processed" / "inventory.parquet")
    except:
        inv = pd.read_csv(DATA_DIR / "raw" / "inventory.csv")
        
    try:
        sales = pd.read_parquet(DATA_DIR / "processed" / "sales.parquet")
    except:
        sales = pd.read_csv(DATA_DIR / "raw" / "sales.csv")
        
    inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'])
    inv['year'] = inv['snapshot_date'].dt.year
    inv['month'] = inv['snapshot_date'].dt.month
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    # 1. Calculate Procurement Profile
    monthly_inbound = inv.groupby(['year', 'month'])['units_received'].sum().reset_index()
    yearly_inbound = monthly_inbound.groupby('year')['units_received'].sum().reset_index()
    yearly_inbound.columns = ['year', 'year_total']
    
    merged = pd.merge(monthly_inbound, yearly_inbound, on='year')
    merged['inbound_share'] = merged['units_received'] / (merged['year_total'] + 1e-6)
    
    profile = merged.groupby('month')['inbound_share'].agg(['median', 'mean', 'std']).reset_index()
    
    print("\n--- HISTORICAL PROCUREMENT PROFILE (Share of Annual Inbound) ---")
    print(profile.to_string(index=False))
    
    # 2. Verify Correlation
    monthly_sales = sales.groupby(['year', 'month'])['Revenue'].sum().reset_index()
    corr_df = pd.merge(monthly_inbound, monthly_sales, on=['year', 'month'])
    
    # Calculate Pearson and Spearman correlation
    pearson_corr = corr_df['units_received'].corr(corr_df['Revenue'], method='pearson')
    spearman_corr = corr_df['units_received'].corr(corr_df['Revenue'], method='spearman')
    
    print(f"\n--- CORRELATION: Inbound vs Revenue ---")
    print(f"Pearson (Linear): {pearson_corr:.3f}")
    print(f"Spearman (Rank):  {spearman_corr:.3f}")
    
    # 3. Check stability over recent years (2019-2022) to see if procurement cycle changed after collapse
    recent_merged = merged[merged['year'] >= 2019]
    recent_profile = recent_merged.groupby('month')['inbound_share'].median().reset_index()
    recent_profile.columns = ['month', 'recent_median']
    
    comparison = pd.merge(profile[['month', 'median']], recent_profile, on='month')
    comparison.columns = ['Month', 'All-Time Median', 'Post-2019 Median']
    
    print("\n--- PROFILE STABILITY (All-time vs Post-Collapse) ---")
    print(comparison.to_string(index=False))

if __name__ == '__main__':
    analyze_procurement()
