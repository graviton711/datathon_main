import pandas as pd
import numpy as np

def analyze_growth_transfer():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 1. Calculate 2021 Momentum (Q4 2021 / Q4 2020)
    q4_2020 = sales[(sales['Date'].dt.year == 2020) & (sales['Date'].dt.month >= 10)]['Revenue'].sum()
    q4_2021 = sales[(sales['Date'].dt.year == 2021) & (sales['Date'].dt.month >= 10)]['Revenue'].sum()
    mom_2021 = (q4_2021 / q4_2020) - 1
    
    print(f"2021 Momentum (Q4 2021 vs Q4 2020): {mom_2021:.2%}")
    
    # 2. Compare 2022 vs 2021 daily/weekly/monthly
    df_21 = sales[sales['Date'].dt.year == 2021].copy()
    df_22 = sales[sales['Date'].dt.year == 2022].copy()
    
    df_21['day_of_year'] = df_21['Date'].dt.dayofyear
    df_22['day_of_year'] = df_22['Date'].dt.dayofyear
    
    # Merge on day_of_year to compare YoY growth
    merged = pd.merge(df_22, df_21, on='day_of_year', suffixes=('_2022', '_2021'))
    merged['yoy_growth'] = (merged['Revenue_2022'] / merged['Revenue_2021']) - 1
    
    print("\n=== 2022 YoY GROWTH ANALYSIS ===")
    overall_growth = (df_22['Revenue'].sum() / df_21['Revenue'].sum()) - 1
    print(f"Overall 2022/2021 Growth: {overall_growth:.2%}")
    
    # Monthly Growth
    merged['month'] = merged['Date_2022'].dt.month
    monthly_growth = merged.groupby('month')['yoy_growth'].mean()
    print("\nMonthly Average Growth Ratios:")
    for m, g in monthly_growth.items():
        print(f"Month {m:>2}: {g:.2%}")
        
    # Stability of the transfer
    print(f"\nRatio (Overall Growth / 2021 Momentum): {overall_growth / mom_2021:.4f}")
    
    # 3. Calculate 2022 Momentum (Q4 2022 / Q4 2021)
    q4_2022 = sales[(sales['Date'].dt.year == 2022) & (sales['Date'].dt.month >= 10)]['Revenue'].sum()
    mom_2022 = (q4_2022 / q4_2021) - 1
    print(f"\n2022 Momentum (Q4 2022 vs Q4 2021): {mom_2022:.2%}")
    
    # Prediction for 2023 scale
    pred_2023_growth = mom_2022 * (overall_growth / mom_2021)
    print(f"Predicted 2023 Growth (based on transfer rule): {pred_2023_growth:.2%}")

if __name__ == "__main__":
    analyze_growth_transfer()
