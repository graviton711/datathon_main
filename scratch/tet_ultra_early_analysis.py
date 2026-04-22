import pandas as pd
import numpy as np

def analyze_ultra_early_tet():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    tet_dates = {
        2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31',
        2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
        2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
        2021: '2021-02-12', 2022: '2022-02-01'
    }
    
    # Analyze only "Normal" years (Exclude 2020, 2021, 2022)
    normal_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    
    daily_impact = {i: [] for i in range(-90, 15)}
    
    for year in normal_years:
        if year not in tet_dates: continue
        tet_date = pd.to_datetime(tet_dates[year])
        
        # Use a very wide baseline (90 days before to 30 days after)
        start_baseline = tet_date - pd.Timedelta(days=90)
        end_baseline = tet_date + pd.Timedelta(days=30)
        
        df_local = sales[(sales['Date'] >= start_baseline) & (sales['Date'] <= end_baseline)].copy()
        avg_local = df_local['Revenue'].mean()
        if avg_local == 0: continue
        
        df_local['days_to_tet'] = (df_local['Date'] - tet_date).dt.days
        
        for d in range(-90, 15):
            val = df_local[df_local['days_to_tet'] == d]['Revenue']
            if not val.empty:
                daily_impact[d].append(val.values[0] / avg_local)
                
    print("=== ULTRA EARLY TET ANALYSIS (T-90 to T+14) - NORMAL YEARS ONLY ===")
    
    results = []
    for d in range(-90, 15):
        mean_ratio = np.mean(daily_impact[d]) if daily_impact[d] else np.nan
        results.append((d, mean_ratio))
        
    # Group by weeks to see the trend
    print("\nWeekly Trend (7-day window average):")
    for w in range(-12, 3): # Week -12 to +2
        days = range(w*7, (w+1)*7)
        vals = [r[1] for r in results if r[0] in days]
        avg_w = np.nanmean(vals) if vals else np.nan
        bar = "#" * int(avg_w * 10) if not np.isnan(avg_w) else ""
        print(f"Week {w:>3} (T{w*7:+} to T{(w+1)*7-1:+}): {avg_w:.2f}x | {bar}")

if __name__ == "__main__":
    analyze_ultra_early_tet()
