import pandas as pd
import numpy as np

def analyze_tet_impact_extended():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    tet_dates = {
        2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31',
        2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
        2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
        2021: '2021-02-12', 2022: '2022-02-01'
    }
    
    daily_impact = {i: [] for i in range(-45, 15)}
    
    for year, tet_str in tet_dates.items():
        if year not in sales['Date'].dt.year.unique(): continue
            
        tet_date = pd.to_datetime(tet_str)
        # Use only Q1 + Dec of previous year to get a LOCAL baseline, avoiding Q4 Mega Sale distortion
        start_date = tet_date - pd.Timedelta(days=60)
        end_date = tet_date + pd.Timedelta(days=30)
        
        df_local = sales[(sales['Date'] >= start_date) & (sales['Date'] <= end_date)].copy()
        avg_local = df_local['Revenue'].mean()
        
        if avg_local == 0: continue
            
        df_local['days_to_tet'] = (df_local['Date'] - tet_date).dt.days
        
        for d in range(-45, 15):
            val = df_local[df_local['days_to_tet'] == d]['Revenue']
            if not val.empty:
                daily_impact[d].append(val.values[0] / avg_local)
                
    print("=== EXTENDED TET IMPACT (T-45 to T+14) vs LOCAL BASELINE ===")
    
    results = []
    for d in range(-45, 15):
        mean_ratio = np.mean(daily_impact[d]) if daily_impact[d] else np.nan
        results.append((d, mean_ratio))
        
    def print_phase(name, phase_data):
        print(f"\n--- {name} ---")
        avg_phase = np.nanmean([x[1] for x in phase_data])
        for d, r in phase_data:
            bar = "#" * int(r * 10)
            print(f"T{d:+} (Day {d:>3}): {r:.2f}x  | {bar}")
        print(f">> Avg Multiplier: {avg_phase:.2f}x")

    print_phase("Early Shopping (T-45 to T-22)", [r for r in results if -45 <= r[0] <= -22])
    print_phase("Main Build-up (T-21 to T-8)", [r for r in results if -21 <= r[0] <= -8])
    print_phase("Pre-Tet Rush & Drop (T-7 to T-1)", [r for r in results if -7 <= r[0] <= -1])
    print_phase("Tet & Post-Tet (T+0 to T+14)", [r for r in results if 0 <= r[0] <= 14])

if __name__ == "__main__":
    analyze_tet_impact_extended()
