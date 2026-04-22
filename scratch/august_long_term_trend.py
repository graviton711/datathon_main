import pandas as pd
import numpy as np

def analyze_august_long_term():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    results = []
    for year in range(2018, 2023):
        df_year = sales[sales['Date'].dt.year == year]
        if df_year.empty: continue
        
        total_rev = df_year['Revenue'].sum()
        aug_rev = df_year[df_year['Date'].dt.month == 8]['Revenue'].sum()
        
        # Calculate weight of August in the total year
        weight = aug_rev / total_rev
        
        # Also check Aug vs July (Seasonal Momentum)
        jul_rev = df_year[df_year['Date'].dt.month == 7]['Revenue'].sum()
        momentum = aug_rev / jul_rev if jul_rev > 0 else 0
        
        results.append({
            'Year': year,
            'Aug_Weight_of_Year': f"{weight:.2%}",
            'Aug_vs_Jul_Ratio': f"{momentum:.2f}x"
        })
        
    print("=== AUGUST SEASONAL STABILITY (2018-2022) ===")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    analyze_august_long_term()
