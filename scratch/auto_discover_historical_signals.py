import pandas as pd
import numpy as np

def auto_discover():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['month'] = sales['Date'].dt.month
    sales['day'] = sales['Date'].dt.day
    sales['year'] = sales['Date'].dt.year
    
    # 1. Discover all days with consistent historical lift (> 1.2x)
    print("Scanning all calendar dates for historical signals (2012-2022)...")
    signals = []
    for m in range(1, 13):
        for d in range(1, 32):
            lifts = []
            for yr in range(2012, 2023):
                target = sales[(sales['year'] == yr) & (sales['month'] == m) & (sales['day'] == d)]
                if target.empty: continue
                
                # Baseline: Avg of that month excluding a window around this day
                window_range = pd.date_range(target['Date'].iloc[0] - pd.Timedelta(days=10), 
                                             target['Date'].iloc[0] + pd.Timedelta(days=10))
                month_mask = (sales['year'] == yr) & (sales['month'] == m)
                baseline = sales[month_mask & (~sales['Date'].isin(window_range))]['Revenue'].mean()
                
                if baseline > 0:
                    lifts.append(target['Revenue'].values[0] / baseline)
            
            if len(lifts) >= 5: # Must have at least 5 years of data
                median_l = np.median(lifts)
                if median_l > 1.2:
                    signals.append({'month': m, 'day': d, 'median_lift': median_l, 'years': len(lifts)})
    
    signals_df = pd.DataFrame(signals).sort_values('median_lift', ascending=False)
    
    # 2. Compare with our current discrepancies
    curr = pd.read_csv('submissions/submission.csv')
    best = pd.read_csv('data/best_submit/best_750k.csv')
    df_comp = curr.merge(best, on='Date', suffixes=('_curr', '_best'))
    df_comp['Date'] = pd.to_datetime(df_comp['Date'])
    df_comp['error'] = df_comp['Revenue_best'] - df_comp['Revenue_curr']
    df_comp['month'] = df_comp['Date'].dt.month
    df_comp['day'] = df_comp['Date'].dt.day
    
    # Merge signals with discrepancies
    result = signals_df.merge(df_comp.groupby(['month', 'day'])['error'].mean().reset_index(), on=['month', 'day'])
    result = result.sort_values('error', ascending=False)
    
    print("\n=== TOP HISTORICAL SIGNALS WITH LARGE REMAINING ERRORS ===")
    print(result.head(20).to_string(index=False))

if __name__ == "__main__":
    auto_discover()
