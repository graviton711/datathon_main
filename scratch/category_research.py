import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_category_evolution():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    products = pd.read_parquet(DATA_DIR / "products.parquet")[['product_id', 'category']]
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date']]
    
    # 2. Link Revenue to Category
    items['item_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
    df = pd.merge(items, products, on='product_id')
    df = pd.merge(df, orders, on='order_id')
    df['year'] = pd.to_datetime(df['order_date']).dt.year
    
    # 3. Yearly Category Share
    cat_yearly = df.groupby(['year', 'category'])['item_rev'].sum().unstack(fill_value=0)
    cat_share = cat_yearly.div(cat_yearly.sum(axis=1), axis=0)
    
    print("--- Category Revenue Share Over Years ---")
    print(cat_share)
    
    # 4. Growth Rates per Category (2021 vs 2022)
    growth_21_22 = cat_yearly.loc[2022] / cat_yearly.loc[2021] - 1.0
    print("\n--- Category Growth Rate (2022 vs 2021) ---")
    print(growth_21_22.sort_values(ascending=False))
    
    # 5. Plotting
    cat_share.plot(kind='area', stacked=True, figsize=(15, 7), alpha=0.7)
    plt.title("Category Mix Evolution (2012-2022)")
    plt.ylabel("Revenue Share")
    plt.savefig("e:/VSCODE_WORKSPACE/NewDatathon/data/plots/category_evolution.png")

if __name__ == "__main__":
    analyze_category_evolution()
