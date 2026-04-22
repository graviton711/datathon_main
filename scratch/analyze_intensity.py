import pandas as pd
import numpy as np
from src.config import Config

def analyze_peak_intensity():
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Calculate Monthly Baseline
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    monthly_mean = sales.groupby(['year', 'month'])['Revenue'].mean().reset_index()
    monthly_mean.columns = ['year', 'month', 'monthly_avg']
    
    df = pd.merge(sales, monthly_mean, on=['year', 'month'])
    df['intensity'] = df['Revenue'] / (df['monthly_avg'] + 1e-6)
    
    # Major peaks to check
    peaks = [
        (10, 10), (11, 11), (12, 12), # Double days
        (5, 1), (9, 2), (4, 30)       # Holidays
    ]
    
    results = []
    for m, d in peaks:
        for yr in [2021, 2022]:
            val = df[(df['year'] == yr) & (df['month'] == m) & (df['Date'].dt.day == d)]
            if not val.empty:
                results.append({
                    'Event': f"{d}/{m}",
                    'Year': yr,
                    'Intensity': val['intensity'].values[0],
                    'Revenue': val['Revenue'].values[0]
                })
    
    res_df = pd.DataFrame(results)
    pivot = res_df.pivot(index='Event', columns='Year', values='Intensity')
    pivot['Growth'] = pivot[2022] / pivot[2021]
    
    print("=== PEAK INTENSITY RECOVERY (2022 vs 2021) ===")
    print(pivot)

if __name__ == "__main__":
    analyze_peak_intensity()
