import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_demographics():
    # 1. Load Data
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date', 'customer_id']]
    orders['Date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['Date'].dt.year
    
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")[['customer_id', 'gender', 'age_group']]
    
    # 2. Join
    df = pd.merge(orders, customers, on='customer_id', how='left')
    
    # 3. Analyze Age Group Share over years
    age_share = df.groupby(['year', 'age_group']).size().unstack(fill_value=0)
    age_share = age_share.div(age_share.sum(axis=1), axis=0)
    
    print("--- Age Group Share of Orders Over Years ---")
    print(age_share)
    
    # 4. Analyze Gender Share
    gender_share = df.groupby(['year', 'gender']).size().unstack(fill_value=0)
    gender_share = gender_share.div(gender_share.sum(axis=1), axis=0)
    
    print("\n--- Gender Share of Orders Over Years ---")
    print(gender_share)
    
    # 5. Link with Categories (Latest data 2021-2022)
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    products = pd.read_parquet(DATA_DIR / "products.parquet")[['product_id', 'category']]
    items = pd.merge(items, products, on='product_id')
    
    order_cat = items.groupby(['order_id', 'category']).size().reset_index().rename(columns={0: 'count'})
    # Join with age info
    df_cat = pd.merge(df, order_cat, on='order_id')
    
    # Category preference by age group in 2022
    cat_by_age = df_cat[df_cat['year'] == 2022].groupby(['age_group', 'category']).size().unstack(fill_value=0)
    cat_by_age_norm = cat_by_age.div(cat_by_age.sum(axis=1), axis=0)
    
    print("\n--- Category Preference by Age Group (2022) ---")
    print(cat_by_age_norm)
    
    # 6. Visualizing
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    age_share.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='RdYlBu')
    plt.title("Age Group Mix Shift")
    
    plt.subplot(1, 2, 2)
    cat_by_age_norm.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
    plt.title("Category Preference by Age")
    
    plt.tight_layout()
    plt.savefig("e:/VSCODE_WORKSPACE/NewDatathon/data/plots/demographics_shift.png")

if __name__ == "__main__":
    analyze_demographics()
