import pandas as pd
import numpy as np

def deep_dive_august_2022():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Filter for August 2021 and 2022
    aug_21 = sales[(sales['Date'].dt.year == 2021) & (sales['Date'].dt.month == 8)].copy()
    aug_22 = sales[(sales['Date'].dt.year == 2022) & (sales['Date'].dt.month == 8)].copy()
    
    print(f"Total Revenue Aug 2021: {aug_21['Revenue'].sum():,.0f}")
    print(f"Total Revenue Aug 2022: {aug_22['Revenue'].sum():,.0f}")
    print(f"YoY Growth: {(aug_22['Revenue'].sum() / aug_21['Revenue'].sum()) - 1:.2%}")
    
    # Check for daily peaks in Aug 2022
    print("\nTop 5 Days in August 2022:")
    print(aug_22.sort_values('Revenue', ascending=False).head(5)[['Date', 'Revenue']])
    
    # Check for 8/8 Sale impact
    s88_22 = aug_22[aug_22['Date'] == '2022-08-08']['Revenue'].values[0]
    avg_22 = aug_22['Revenue'].mean()
    print(f"\n8/8/2022 Revenue: {s88_22:,.0f} ({s88_22/avg_22:.2f}x of Aug avg)")
    
    # Check context: Compare Aug 2021 to July 2021 and Sep 2021
    # This helps see if Aug 2021 was a "dip"
    jul_21 = sales[(sales['Date'].dt.year == 2021) & (sales['Date'].dt.month == 7)]['Revenue'].sum()
    sep_21 = sales[(sales['Date'].dt.year == 2021) & (sales['Date'].dt.month == 9)]['Revenue'].sum()
    
    print("\n--- 2021 Context ---")
    print(f"July 2021: {jul_21:,.0f}")
    print(f"Aug 2021 : {aug_21['Revenue'].sum():,.0f} (Dip: {aug_21['Revenue'].sum()/jul_21 - 1:.2%})")
    print(f"Sep 2021 : {sep_21:,.0f}")
    
    # Compare Aug 2022 to Jul 2022
    jul_22 = sales[(sales['Date'].dt.year == 2022) & (sales['Date'].dt.month == 7)]['Revenue'].sum()
    print("\n--- 2022 Context ---")
    print(f"July 2022: {jul_22:,.0f}")
    print(f"Aug 2022 : {aug_22['Revenue'].sum():,.0f} (Jump: {aug_22['Revenue'].sum()/jul_22 - 1:.2%})")

if __name__ == "__main__":
    deep_dive_august_2022()
