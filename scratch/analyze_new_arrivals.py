import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
ORDER_ITEMS_FILE = DATA_DIR / "order_items.parquet"
ORDERS_FILE = DATA_DIR / "orders.parquet"

def analyze_new_arrivals():
    print("Loading data for new arrival research...")
    orders = pd.read_parquet(ORDERS_FILE)[['order_id', 'order_date']]
    items = pd.read_parquet(ORDER_ITEMS_FILE)[['order_id', 'product_id']]
    
    # Merge to get dates for each product sale
    df = pd.merge(items, orders, on='order_id')
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Find first sale date for each product
    product_birth = df.groupby('product_id')['order_date'].min().reset_index()
    product_birth.columns = ['product_id', 'birth_date']
    
    # Aggregate by month
    product_birth['birth_month'] = product_birth['birth_date'].dt.month
    product_birth['birth_year'] = product_birth['birth_date'].dt.year
    
    launch_stats = product_birth.groupby(['birth_year', 'birth_month']).size().reset_index(name='new_product_count')
    
    # Merge with total revenue
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year_tmp'] = sales['Date'].dt.year
    sales['month_tmp'] = sales['Date'].dt.month
    monthly_sales = sales.groupby(['year_tmp', 'month_tmp'])['Revenue'].sum().reset_index()
    monthly_sales.columns = ['year', 'month', 'Revenue']
    
    final = pd.merge(monthly_sales, launch_stats, left_on=['year', 'month'], right_on=['birth_year', 'birth_month'], how='left').fillna(0)
    
    # Correlation
    corr = final['new_product_count'].corr(final['Revenue'])
    print(f"\n--- Correlation: New Product Launches vs Revenue ---")
    print(f"Correlation: {corr:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(final['new_product_count'], final['Revenue'])
    plt.xlabel('New Products Launched')
    plt.ylabel('Monthly Revenue')
    plt.title(f'Impact of New Arrivals on Revenue (Corr: {corr:.4f})')
    plt.savefig('scratch/new_arrivals_impact.png')
    print("\nAnalysis saved to scratch/new_arrivals_impact.png")

if __name__ == "__main__":
    analyze_new_arrivals()
