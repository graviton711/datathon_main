import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def analyze_inertia_weights():
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    orders = pd.read_parquet(Config.ORDERS_FILE)
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    # 2. Daily Aggregation
    daily_rev = sales.groupby('Date')['Revenue'].sum()
    daily_orders = orders.groupby('order_date')['order_id'].count()
    
    # Ensure indices are aligned
    all_dates = pd.date_range(start=daily_rev.index.min(), end=daily_rev.index.max())
    daily_rev = daily_rev.reindex(all_dates, fill_value=0)
    daily_orders = daily_orders.reindex(all_dates, fill_value=0)
    daily_aov = daily_rev / (daily_orders + 1e-6)
    
    data = pd.DataFrame({
        'revenue': daily_rev,
        'orders': daily_orders,
        'aov': daily_aov
    })
    data['year'] = data.index.year
    data['month'] = data.index.month
    
    # 3. Calculate Q4 Totals and Annual Medians
    results = []
    years = sorted(data['year'].unique())
    
    for yr in years:
        # Annual Median (excluding potential outliers)
        annual_median = data[data['year'] == yr]['revenue'].median()
        
        # Q4 Totals
        q4_mask = (data['year'] == yr) & (data['month'] >= 10)
        q4_rev = data[q4_mask]['revenue'].sum()
        q4_orders = data[q4_mask]['orders'].sum()
        q4_aov = data[q4_mask]['aov'].median()
        
        results.append({
            'year': yr,
            'annual_median': annual_median,
            'q4_rev': q4_rev,
            'q4_orders': q4_orders,
            'q4_aov': q4_aov
        })
        
    df_yr = pd.DataFrame(results)
    
    # 4. Calculate YoY Momentum and Next Year Growth
    analysis_rows = []
    for i in range(2, len(df_yr) - 1): # Need N-2, N-1, and N+1
        # Year N is the forecast-origin year
        # We look at Q4 of Year N-1 vs Q4 of Year N-2
        # And see how it affects Year N
        yr_curr = df_yr.iloc[i]['year']
        
        # Target: Realized Growth for Year N
        realized_g = df_yr.iloc[i]['annual_median'] / (df_yr.iloc[i-1]['annual_median'] + 1e-6)
        
        # Signals: Q4 YoY Momentum (N-1 vs N-2)
        q4_m_rev = df_yr.iloc[i-1]['q4_rev'] / (df_yr.iloc[i-2]['q4_rev'] + 1e-6)
        q4_m_orders = df_yr.iloc[i-1]['q4_orders'] / (df_yr.iloc[i-2]['q4_orders'] + 1e-6)
        q4_m_aov = df_yr.iloc[i-1]['q4_aov'] / (df_yr.iloc[i-2]['q4_aov'] + 1e-6)
        
        analysis_rows.append({
            'target_year': yr_curr,
            'realized_growth': realized_g,
            'm_rev': q4_m_rev,
            'm_orders': q4_m_orders,
            'm_aov': q4_m_aov
        })
        
    analysis_df = pd.DataFrame(analysis_rows)
    print("\n--- Historical Momentum Analysis ---")
    print(analysis_df)
    
    if len(analysis_df) >= 2:
        corrs = analysis_df[['realized_growth', 'm_rev', 'm_orders', 'm_aov']].corr()['realized_growth'].drop('realized_growth')
        print("\n--- Correlation with Realized Growth ---")
        print(corrs)
        
        # Simple attribution weights (Normalized absolute correlation)
        weights = corrs.abs() / corrs.abs().sum()
        print("\n--- Data-Driven Weights Recommendation ---")
        print(weights)
    else:
        print("\nInsufficient data points for robust weighting. Need at least 3 years of Q4 history.")

if __name__ == "__main__":
    analyze_inertia_weights()
