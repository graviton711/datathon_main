import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("data/processed")
INVENTORY_FILE = DATA_DIR / "inventory.parquet"
SALES_FILE = DATA_DIR / "sales.parquet"

def analyze_inventory_impact():
    print("Loading data for inventory impact research...")
    inv = pd.read_parquet(INVENTORY_FILE)
    sales = pd.read_parquet(SALES_FILE)
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    # 1. Monthly Category Inventory Stats
    # Aggregate stockout_days and fill_rate by category
    inv_monthly = inv.groupby(['year', 'month', 'category']).agg({
        'stockout_days': 'mean',
        'fill_rate': 'mean',
        'stock_on_hand': 'sum'
    }).reset_index()
    
    # Calculate Global Stockout Severity (Weighted by Streetwear since it's 80% of rev)
    streetwear_inv = inv_monthly[inv_monthly['category'] == 'Streetwear'].copy()
    
    # 2. Monthly Revenue
    monthly_sales = sales.groupby(['year', 'month'])['Revenue'].sum().reset_index()
    
    # 3. Merge
    df = pd.merge(monthly_sales, streetwear_inv, on=['year', 'month'], how='inner')
    
    # Calculate Revenue Efficiency (Revenue / Stock_on_Hand) - how well we sell what we have
    df['rev_per_stock'] = df['Revenue'] / (df['stock_on_hand'] + 1e-6)
    
    # 4. Correlation Analysis
    corr_matrix = df[['Revenue', 'stockout_days', 'fill_rate', 'stock_on_hand', 'rev_per_stock']].corr()
    print("\n--- Correlation: Inventory Metrics vs Revenue (Monthly) ---")
    print(corr_matrix['Revenue'].sort_values())
    
    # 5. Visualization: Stockout Days vs Revenue
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='stockout_days', y='Revenue')
    plt.title('Impact of Streetwear Stockout Days on Total Revenue')
    plt.savefig('scratch/inventory_impact_analysis.png')
    
    # 6. Time Series View
    df['month_dt'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['month_dt'], df['Revenue'], label='Revenue', color='blue')
    plt.title('Monthly Revenue')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['month_dt'], df['stockout_days'], label='Avg Stockout Days (Streetwear)', color='red')
    plt.title('Streetwear Stockout Intensity')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('scratch/inventory_trend_analysis.png')
    print("\nAnalysis saved to scratch/inventory_trend_analysis.png")

if __name__ == "__main__":
    analyze_inventory_impact()
