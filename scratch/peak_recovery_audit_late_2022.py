import pandas as pd
import numpy as np

def audit_peak_recovery():
    # 1. Load Data
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Identify Double Days (10/10, 11/11, 12/12)
    double_days = [
        (10, 10),
        (11, 11),
        (12, 12)
    ]
    
    results = []
    for m, d in double_days:
        for yr in [2021, 2022]:
            peak_date = pd.to_datetime(f"{yr}-{m:02d}-{d:02d}")
            if peak_date not in sales['Date'].values: continue
            
            peak_rev = sales[sales['Date'] == peak_date]['Revenue'].sum()
            # Monthly baseline excluding the peak day
            month_mask = (sales['Date'].dt.year == yr) & (sales['Date'].dt.month == m)
            baseline = sales[month_mask & (sales['Date'].dt.day != d)]['Revenue'].mean()
            
            lift = peak_rev / baseline
            results.append({
                'Event': f"{m}/{d}",
                'Year': yr,
                'Revenue': peak_rev,
                'Lift': lift
            })
            
    df = pd.DataFrame(results)
    print("=== LATE 2022 DOUBLE-DAY PEAK RECOVERY AUDIT ===")
    print(df)
    
    # 3. Calculate Recovery Ratio (2022 Lift / 2021 Lift)
    pivot = df.pivot(index='Event', columns='Year', values='Lift')
    pivot['Recovery_Factor'] = pivot[2022] / pivot[2021]
    print("\n=== PEAK INTENSITY RECOVERY FACTOR (2022 vs 2021) ===")
    print(pivot)

if __name__ == "__main__":
    audit_peak_recovery()
