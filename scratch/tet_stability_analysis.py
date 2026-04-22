import pandas as pd
import numpy as np

def check_tet_stability():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    tet_dates = {
        2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31',
        2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
        2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
        2021: '2021-02-12', 2022: '2022-02-01'
    }
    
    phases = {
        'Early Shop (T-35 to T-25)': (-35, -25),
        'Tet Rush (T-7 to T-1)': (-7, -1),
        'Tet Holiday (T0 to T+5)': (0, 5),
        'Recovery (T+6 to T+14)': (6, 14)
    }
    
    stability_data = {phase: [] for phase in phases}
    
    for year, tet_str in tet_dates.items():
        if year not in sales['Date'].dt.year.unique(): continue
        
        tet_date = pd.to_datetime(tet_str)
        start_baseline = tet_date - pd.Timedelta(days=60)
        end_baseline = tet_date + pd.Timedelta(days=30)
        
        df_local = sales[(sales['Date'] >= start_baseline) & (sales['Date'] <= end_baseline)].copy()
        avg_local = df_local['Revenue'].mean()
        if avg_local == 0: continue
        
        df_local['days_to_tet'] = (df_local['Date'] - tet_date).dt.days
        
        for phase_name, (start, end) in phases.items():
            phase_rev = df_local[(df_local['days_to_tet'] >= start) & (df_local['days_to_tet'] <= end)]['Revenue'].mean()
            stability_data[phase_name].append({
                'Year': year,
                'Lift': phase_rev / avg_local
            })
            
    print("=== TET PHASE STABILITY ACROSS YEARS ===")
    for phase_name, data in stability_data.items():
        df_p = pd.DataFrame(data)
        mean_l = df_p['Lift'].mean()
        std_l = df_p['Lift'].std()
        cv_l = std_l / mean_l
        
        print(f"\n>>> {phase_name}")
        print(df_p.to_string(index=False, formatters={'Lift': '{:.3f}x'.format}))
        print(f"Full Stats: Mean={mean_l:.3f}x, Std={std_l:.3f}, CV={cv_l:.3f}")
        
        # Filtered Stats (Exclude 2020, 2021, 2022)
        df_f = df_p[df_p['Year'] < 2020]
        f_mean = df_f['Lift'].mean()
        f_std = df_f['Lift'].std()
        f_cv = f_std / f_mean
        print(f"Normal Years (Pre-2020) Stats: Mean={f_mean:.3f}x, Std={f_std:.3f}, CV={f_cv:.3f}")
        
        # Check for outliers in full data
        outliers = df_p[np.abs(df_p['Lift'] - mean_l) > 2 * std_l]
        if not outliers.empty:
            print(f"!! Potential Outliers detected: {outliers['Year'].tolist()}")

if __name__ == "__main__":
    check_tet_stability()
