import pandas as pd
import numpy as np

def audit_peak_amplitude():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Analyze May 1st (Labor Day) across years
    may1 = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day == 1)]
    print("=== MAY 1ST HISTORICAL REVENUE ===")
    print(may1[['Date', 'Revenue']])
    
    # Analyze August 1st-4th
    aug_peak = sales[(sales['Date'].dt.month == 8) & (sales['Date'].dt.day <= 4)]
    print("\n=== AUGUST 1-4 HISTORICAL REVENUE (AVG) ===")
    print(aug_peak.groupby(aug_peak['Date'].dt.year)['Revenue'].mean())

if __name__ == "__main__":
    audit_peak_amplitude()
