import pandas as pd
import numpy as np

def analyze_category_growth():
    # Load data
    items = pd.read_parquet('data/processed/order_items.parquet')
    products = pd.read_parquet('data/processed/products.parquet')
    orders = pd.read_parquet('data/processed/orders.parquet')
    
    # Pre-process
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['order_date'].dt.year
    
    # Merge
    df = pd.merge(items, products[['product_id', 'category']], on='product_id')
    df = pd.merge(df, orders[['order_id', 'year', 'order_date']], on='order_id')
    
    # Calculate Revenue
    df['revenue'] = (df['unit_price'] * df['quantity']) - df['discount_amount']
    
    # Calculate Monthly Revenue by Category
    df['month'] = df['order_date'].dt.month
    cat_monthly = df.groupby(['year', 'month', 'category'])['revenue'].sum().reset_index()
    
    # Compare 2022 vs 2021
    y21 = cat_monthly[cat_monthly['year'] == 2021].groupby('category')['revenue'].sum()
    y22 = cat_monthly[cat_monthly['year'] == 2022].groupby('category')['revenue'].sum()
    
    growth = (y22 / y21).sort_values(ascending=False).reset_index(name='growth_22_vs_21')
    
    print("=== CATEGORY GROWTH (2022 vs 2021) ===")
    print(growth.head(10))
    
    # Check what was hot in Q4 2022
    q4_22 = cat_monthly[(cat_monthly['year'] == 2022) & (cat_monthly['month'] >= 10)]
    q4_21 = cat_monthly[(cat_monthly['year'] == 2021) & (cat_monthly['month'] >= 10)]
    
    q4_22_rev = q4_22.groupby('category')['revenue'].sum()
    q4_21_rev = q4_21.groupby('category')['revenue'].sum()
    
    q4_growth = (q4_22_rev / q4_21_rev).sort_values(ascending=False).reset_index(name='growth_q4_22_vs_q4_21')
    print("\n=== Q4 MOMENTUM BY CATEGORY (Q4 22 vs Q4 21) ===")
    print(q4_growth.head(10))

if __name__ == "__main__":
    analyze_category_growth()
