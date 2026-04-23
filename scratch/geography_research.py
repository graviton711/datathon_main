import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_geography():
    # 1. Load Data
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date', 'zip']]
    orders['Date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['Date'].dt.year
    orders['month'] = orders['Date'].dt.month
    
    geo = pd.read_parquet(DATA_DIR / "geography.parquet")[['zip', 'region']]
    
    # 2. Join
    df = pd.merge(orders, geo, on='zip', how='left')
    
    # Get Revenue per order (approximate from sales/orders ratio or just use order count as proxy)
    # Better: Load sales and map to orders
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # For research, let's use Order Count as a high-fidelity proxy for Revenue volume
    daily_geo = df.groupby(['Date', 'region']).size().unstack(fill_value=0)
    
    # 3. Analyze Share Shift
    yearly_geo = df.groupby(['year', 'region']).size().unstack(fill_value=0)
    yearly_share = yearly_geo.div(yearly_geo.sum(axis=1), axis=0)
    
    print("--- Regional Order Share Over Years ---")
    print(yearly_share)
    
    # 4. Analyze Seasonality Difference (Normalized to each region's median)
    monthly_geo = df.groupby(['month', 'region']).size().unstack(fill_value=0)
    monthly_shape = monthly_geo.div(monthly_geo.median(axis=0), axis=1)
    
    print("\n--- Regional Seasonal Shape (Relative to Region Median) ---")
    print(monthly_shape)
    
    # 5. Visualizing the Difference
    plt.figure(figsize=(12, 6))
    for region in monthly_shape.columns:
        plt.plot(monthly_shape.index, monthly_shape[region], label=region, marker='o')
    plt.title("Regional Seasonality Comparison")
    plt.xlabel("Month")
    plt.ylabel("Multiplier (vs Region Median)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("e:/VSCODE_WORKSPACE/NewDatathon/data/plots/geography_seasonality.png")
    
    # 6. Conclusion
    # If North has a higher Q4 multiplier than South, we found something!
    q4_north = monthly_shape.loc[10:12, 'North'].mean() if 'North' in monthly_shape.columns else 0
    q4_south = monthly_shape.loc[10:12, 'South'].mean() if 'South' in monthly_shape.columns else 0
    
    print(f"\nQ4 Avg Multiplier - North: {q4_north:.3f}, South: {q4_south:.3f}")
    if abs(q4_north - q4_south) > 0.05:
        print("Significant Seasonality Difference Detected!")

if __name__ == "__main__":
    analyze_geography()
