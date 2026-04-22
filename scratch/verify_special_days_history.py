import pandas as pd
import numpy as np

def verify_special_days():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['month'] = sales['Date'].dt.month
    sales['day'] = sales['Date'].dt.day
    sales['year'] = sales['Date'].dt.year
    
    check_days = [
        (10, 20, "VN Women's Day"),
        (7, 1, "July 1 (Start of July)"),
        (12, 31, "New Year's Eve")
    ]
    
    results = []
    for m, d, label in check_days:
        lifts = []
        for yr in range(2018, 2023): # Check recent 5 years
            target = sales[(sales['year'] == yr) & (sales['month'] == m) & (sales['day'] == d)]
            if target.empty: continue
            
            # Baseline: Avg of that month excluding the day itself
            baseline = sales[(sales['year'] == yr) & (sales['month'] == m) & (sales['day'] != d)]['Revenue'].mean()
            if baseline > 0:
                lifts.append(target['Revenue'].values[0] / baseline)
        
        if lifts:
            results.append({
                'Label': label,
                'Avg_Lift': f"{np.mean(lifts):.2f}x",
                'Max_Lift': f"{np.max(lifts):.2f}x",
                'Years_Active': len(lifts)
            })
            
    print("=== HISTORICAL VERIFICATION OF SPECIAL DAYS (2018-2022) ===")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    verify_special_days()
