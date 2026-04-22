import pandas as pd
import numpy as np

def audit_momentum():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    
    yearly_rev = sales.groupby('year')['Revenue'].sum()
    print("=== YEARLY REVENUE & GROWTH ===")
    growth = yearly_rev.pct_change()
    print(pd.DataFrame({'Revenue': yearly_rev, 'YoY_Growth': growth}))
    
    # Analyze the 'Acceleration' (Growth of Growth)
    acceleration = growth.diff()
    print("\n=== GROWTH ACCELERATION ===")
    print(acceleration)

if __name__ == "__main__":
    audit_momentum()
