import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
INVENTORY_FILE = DATA_DIR / "inventory.parquet"
SALES_FILE = DATA_DIR / "sales.parquet"

def analyze_inventory_turnover():
    print("Analyzing Inventory Turnover Dynamics...")
    inv = pd.read_parquet(INVENTORY_FILE)
    sales = pd.read_parquet(SALES_FILE)
    
    # Calculate Monthly Turnover: Units Sold / Units Received
    # This shows how efficiently we are clearing the new arrivals
    inv_monthly = inv.groupby(['year', 'month']).agg({
        'units_sold': 'sum',
        'units_received': 'sum',
        'stock_on_hand': 'mean'
    }).reset_index()
    
    inv_monthly['turnover_ratio'] = inv_monthly['units_sold'] / (inv_monthly['units_received'] + 1e-6)
    inv_monthly['stock_velocity'] = inv_monthly['units_sold'] / (inv_monthly['stock_on_hand'] + 1e-6)
    
    # Merge with Revenue
    sales['year'] = pd.to_datetime(sales['Date']).dt.year
    sales['month'] = pd.to_datetime(sales['Date']).dt.month
    monthly_sales = sales.groupby(['year', 'month'])['Revenue'].sum().reset_index()
    
    df = pd.merge(monthly_sales, inv_monthly, on=['year', 'month'])
    
    # Correlation
    corrs = df[['Revenue', 'turnover_ratio', 'stock_velocity', 'stock_on_hand']].corr()
    print("\n--- Inventory Turnover Correlation ---")
    print(corrs['Revenue'].sort_values())
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(df['stock_velocity'], df['Revenue'])
    plt.xlabel('Stock Velocity (Units Sold / Stock on Hand)')
    plt.ylabel('Monthly Revenue')
    plt.title(f"Stock Velocity vs Revenue (Corr: {corrs.loc['Revenue', 'stock_velocity']:.4f})")
    plt.savefig('scratch/inventory_turnover_analysis.png')
    print("\nAnalysis saved to scratch/inventory_turnover_analysis.png")

if __name__ == "__main__":
    analyze_inventory_turnover()
