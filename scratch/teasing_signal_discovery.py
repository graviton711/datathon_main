import pandas as pd
import numpy as np

def discover_teasing_signals():
    # 1. Load Data
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    sales = pd.read_parquet('data/processed/sales.parquet')
    
    traffic['date'] = pd.to_datetime(traffic['date'])
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Analyze the 7 days BEFORE major peaks (May 1st, August 1st, etc.)
    target_dates = [
        (5, 1), # May 1st
        (8, 1), # Aug 1st
        (6, 2), # June 2nd
    ]
    
    print("=== TEASING SIGNAL ANALYSIS (Traffic before Peak) ===")
    for m, d in target_dates:
        print(f"\nAnalyzing Peak {m}/{d}:")
        for yr in range(2018, 2023):
            peak_date = pd.to_datetime(f"{yr}-{m:02d}-{d:02d}")
            if peak_date not in traffic['date'].values: continue
            
            # Look at traffic 7 days before
            teasing_window = pd.date_range(peak_date - pd.Timedelta(days=7), peak_date - pd.Timedelta(days=1))
            teasing_traffic = traffic[traffic['date'].isin(teasing_window)]['sessions'].mean()
            
            # Look at traffic 30 days before (baseline)
            baseline_window = pd.date_range(peak_date - pd.Timedelta(days=37), peak_date - pd.Timedelta(days=8))
            baseline_traffic = traffic[traffic['date'].isin(baseline_window)]['sessions'].mean()
            
            lift = teasing_traffic / (baseline_traffic + 1e-6)
            
            # Actual Revenue Lift on Peak Day
            peak_rev = sales[sales['Date'] == peak_date]['Revenue'].sum()
            month_avg_rev = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == m)]['Revenue'].mean()
            rev_lift = peak_rev / (month_avg_rev + 1e-6)
            
            print(f"Year {yr}: Teasing Traffic Lift={lift:.2f}x | Actual Rev Lift={rev_lift:.2f}x")

if __name__ == "__main__":
    discover_teasing_signals()
