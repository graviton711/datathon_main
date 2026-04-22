import pandas as pd

def analyze_growth():
    best = pd.read_csv('data/best_submit/best_750k.csv')
    best['Date'] = pd.to_datetime(best['Date'])
    best['year'] = best['Date'].dt.year
    best['month'] = best['Date'].dt.month
    
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Compare Best Submission (2023-2024) vs 2022 Actuals
    for yr in [2023, 2024]:
        for m in range(1, 13):
            best_val = best[(best['year'] == yr) & (best['month'] == m)]['Revenue'].sum()
            prev_val = sales[(sales['Date'].dt.year == 2022) & (sales['Date'].dt.month == m)]['Revenue'].sum()
            
            if prev_val > 0:
                print(f"Implied Growth {yr} vs 2022 (Month {m}): {best_val/prev_val:.2f}x")

if __name__ == "__main__":
    analyze_growth()
