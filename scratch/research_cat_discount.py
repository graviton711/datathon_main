import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("data/processed")
ORDER_ITEMS_FILE = DATA_DIR / "order_items.parquet"
ORDERS_FILE = DATA_DIR / "orders.parquet"
PRODUCTS_FILE = DATA_DIR / "products.parquet"

def research_cat_discount():
    print("Loading data for category discount research...")
    orders = pd.read_parquet(ORDERS_FILE)[['order_id', 'order_date']]
    items = pd.read_parquet(ORDER_ITEMS_FILE)
    products = pd.read_parquet(PRODUCTS_FILE)[['product_id', 'category']]
    
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    items = pd.merge(items, orders, on='order_id')
    items = pd.merge(items, products, on='product_id')
    
    # Calculate Revenue and Discount per Category-Day
    items['gross_rev'] = items['quantity'] * items['unit_price']
    items['net_rev'] = items['gross_rev'] - items['discount_amount']
    
    daily_cat = items.groupby(['order_date', 'category']).agg({
        'gross_rev': 'sum',
        'discount_amount': 'sum',
        'net_rev': 'sum'
    }).reset_index()
    
    daily_cat['discount_depth'] = daily_cat['discount_amount'] / (daily_cat['gross_rev'] + 1e-6)
    
    # Calculate Lift: (Daily Net Rev) / (Monthly Median Net Rev for that category)
    daily_cat['month'] = daily_cat['order_date'].dt.month
    daily_cat['year'] = daily_cat['order_date'].dt.year
    
    monthly_median = daily_cat.groupby(['year', 'month', 'category'])['net_rev'].transform('median')
    daily_cat['lift'] = daily_cat['net_rev'] / (monthly_median + 1e-6)
    
    # Correlation between Discount Depth and Lift per Category
    print("\n--- Discount Sensitivity per Category (Correlation: Discount Depth vs Lift) ---")
    results = {}
    for cat in daily_cat['category'].unique():
        cat_df = daily_cat[daily_cat['category'] == cat]
        corr = cat_df['discount_depth'].corr(cat_df['lift'])
        results[cat] = corr
        print(f"{cat:12}: {corr:.4f}")
    
    # Plot sensitivity
    plt.figure(figsize=(12, 8))
    sns.lmplot(data=daily_cat, x='discount_depth', y='lift', col='category', col_wrap=2, scatter_kws={'alpha':0.1})
    plt.savefig('scratch/category_discount_sensitivity.png')
    print("\nVisualization saved to scratch/category_discount_sensitivity.png")

if __name__ == "__main__":
    research_cat_discount()
