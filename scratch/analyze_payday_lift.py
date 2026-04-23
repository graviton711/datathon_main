import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
SALES_FILE = PROJECT_ROOT / "data" / "processed" / "sales.parquet"

def analyze_payday_lift():
    print("Loading sales data...")
    df = pd.read_parquet(SALES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    
    # Define payday window: typically 25th to 5th of next month.
    # To keep it simple per month, we consider days >= 25 and days <= 5 as payday window.
    df['is_payday'] = ((df['day'] >= 25) | (df['day'] <= 5))
    
    # Calculate daily average revenue for payday vs non-payday per (year, month)
    monthly_stats = df.groupby(['year', 'month', 'is_payday'])['Revenue'].mean().unstack().reset_index()
    monthly_stats.columns = ['year', 'month', 'non_payday_avg', 'payday_avg']
    
    # Calculate lift: Payday / Non-Payday
    monthly_stats['lift'] = monthly_stats['payday_avg'] / (monthly_stats['non_payday_avg'] + 1e-6)
    
    # Aggregate to find the stable historical lift per month
    # Using median to avoid being skewed by single crazy years (like 2019 collapse)
    final_lift = monthly_stats.groupby('month')['lift'].agg(['median', 'mean', 'std']).reset_index()
    final_lift.columns = ['Month', 'Median_Lift', 'Mean_Lift', 'Lift_Std']
    
    print("\n--- PAYDAY LIFT BY MONTH (2012-2022) ---")
    print(final_lift.to_string(index=False))
    
    print("\n--- Validating Insight ---")
    aug = final_lift[final_lift['Month'] == 8]['Median_Lift'].values[0]
    nov = final_lift[final_lift['Month'] == 11]['Median_Lift'].values[0]
    print(f"August Lift: {aug:.2f}x (Insight claimed ~1.78x)")
    print(f"November Lift: {nov:.2f}x (Insight claimed ~1.11x)")

if __name__ == "__main__":
    analyze_payday_lift()
