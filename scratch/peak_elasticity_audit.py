import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_peak_elasticity():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # May 1st data
    may1 = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day == 1)].copy()
    
    results = []
    for _, row in may1.iterrows():
        yr = row['Date'].year
        peak_rev = row['Revenue']
        # Baseline = average of other days in May
        baseline = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == 5) & (sales['Date'].dt.day != 1)]['Revenue'].mean()
        results.append({'Year': yr, 'Peak': peak_rev, 'Baseline': baseline})
        
    df = pd.DataFrame(results)
    
    # Log-Log Regression to find Elasticity (alpha)
    # log(Peak) = log(k) + alpha * log(Baseline)
    log_peak = np.log(df['Peak']).values.reshape(-1, 1)
    log_baseline = np.log(df['Baseline']).values.reshape(-1, 1)
    
    model = LinearRegression().fit(log_baseline, log_peak)
    alpha = model.coef_[0][0]
    
    print(f"=== PEAK ELASTICITY ANALYSIS (May 1st) ===")
    print(f"Calculated Elasticity (alpha): {alpha:.3f}")
    
    if alpha > 1.0:
        print("Insight: Peaks grow FASTER than the baseline (High Elasticity).")
    else:
        print("Insight: Peaks grow SLOWER than the baseline (Low Elasticity).")

if __name__ == "__main__":
    calculate_peak_elasticity()
