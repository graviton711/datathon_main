import pandas as pd
import numpy as np

def verify_payday_lift():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    sales['day'] = sales['Date'].dt.day
    
    # Define Payday Window (28th to 5th of next month)
    # We will classify each day as 'Payday' or 'Normal'
    payday_days = [28, 29, 30, 31, 1, 2, 3, 4, 5]
    sales['is_payday'] = sales['day'].isin(payday_days)
    
    results = []
    
    # Calculate for each year
    for yr in sorted(sales['year'].unique()):
        df_yr = sales[sales['year'] == yr]
        
        payday_avg = df_yr[df_yr['is_payday']]['Revenue'].mean()
        normal_avg = df_yr[~df_yr['is_payday']]['Revenue'].mean()
        
        lift = payday_avg / normal_avg if normal_avg > 0 else np.nan
        
        results.append({
            'Year': yr,
            'Payday_Avg': payday_avg,
            'Normal_Avg': normal_avg,
            'Lift': lift
        })
        
    df_results = pd.DataFrame(results).dropna()
    
    print("=== PAYDAY LIFT YEARLY ANALYSIS (Days 28-31, 1-5 vs Others) ===")
    print(df_results.to_string(index=False, formatters={
        'Payday_Avg': '{:,.0f}'.format,
        'Normal_Avg': '{:,.0f}'.format,
        'Lift': '{:.3f}x'.format
    }))
    
    mean_lift = df_results['Lift'].mean()
    std_lift = df_results['Lift'].std()
    cv_lift = std_lift / mean_lift
    
    print("\n=== STABILITY METRICS ===")
    print(f"Mean Payday Lift: {mean_lift:.3f}x")
    print(f"Standard Deviation: {std_lift:.4f}")
    print(f"Coefficient of Variation (CV): {cv_lift:.4f}")
    
    # Monthly stability check for the last 3 complete years (2020, 2021, 2022)
    print("\n=== RECENT MONTHLY STABILITY (2020-2022) ===")
    recent = sales[sales['year'].isin([2020, 2021, 2022])]
    monthly_lifts = []
    
    for (yr, mo), group in recent.groupby(['year', 'month']):
        p_avg = group[group['is_payday']]['Revenue'].mean()
        n_avg = group[~group['is_payday']]['Revenue'].mean()
        if n_avg > 0:
            monthly_lifts.append(p_avg / n_avg)
            
    m_mean = np.mean(monthly_lifts)
    m_std = np.std(monthly_lifts)
    m_cv = m_std / m_mean
    print(f"Monthly Mean Lift: {m_mean:.3f}x")
    print(f"Monthly CV: {m_cv:.4f}")

if __name__ == "__main__":
    verify_payday_lift()
