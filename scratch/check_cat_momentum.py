import pandas as pd
import numpy as np
from src.config import Config

def check_category_momentum():
    print("--- Calculating Category-Specific Momentum ---")
    
    # 1. Load Data
    try:
        sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
        items = pd.read_parquet(Config.DATA_DIR / "processed" / "order_items.parquet")
        products = pd.read_parquet(Config.DATA_DIR / "processed" / "products.parquet")
        orders = pd.read_parquet(Config.ORDERS_FILE)[['order_id', 'order_date']]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Merge to get Category Revenue
    items['item_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
    df = pd.merge(items, orders, on='order_id')
    df = pd.merge(df, products, on='product_id')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year'] = df['order_date'].dt.year
    
    # 3. Aggregate Yearly Revenue by Category
    cat_yearly = df.groupby(['year', 'category'])['item_rev'].sum().unstack().fillna(0)
    
    # 4. Calculate YoY Growth (2022 vs 2021)
    if 2021 in cat_yearly.index and 2022 in cat_yearly.index:
        growth = (cat_yearly.loc[2022] / (cat_yearly.loc[2021] + 1e-6))
        
        # Sort by growth rate
        growth_df = pd.DataFrame({
            'Revenue_2021': cat_yearly.loc[2021],
            'Revenue_2022': cat_yearly.loc[2022],
            'YoY_Growth': growth
        }).sort_values('YoY_Growth', ascending=False)
        
        print("\nCategory Momentum (2022 vs 2021):")
        print(growth_df.to_string())
        
        # Calculate Global for comparison
        global_2021 = cat_yearly.loc[2021].sum()
        global_2022 = cat_yearly.loc[2022].sum()
        global_growth = global_2022 / global_2021
        print(f"\nGlobal Growth: {global_growth:.3f}x")
        
        return growth_df
    else:
        print("Required years (2021, 2022) not found in data.")
        return None

if __name__ == "__main__":
    check_category_momentum()
