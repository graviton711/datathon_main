import pandas as pd
import numpy as np

def audit_price_volatility():
    # 1. Load Data
    items = pd.read_parquet('data/processed/order_items.parquet')
    orders = pd.read_parquet('data/processed/orders.parquet')
    
    # 2. Merge to get dates
    df = items.merge(orders[['order_id', 'order_date']], on='order_id')
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # 3. Analyze May 1st Price vs April Avg Price (Top 100 Products)
    top_prods = df.groupby('product_id')['order_id'].count().sort_values(ascending=False).head(100).index
    
    results = []
    for yr in range(2013, 2023):
        peak_date = pd.to_datetime(f"{yr}-05-01")
        april_mask = (df['order_date'].dt.year == yr) & (df['order_date'].dt.month == 4)
        
        peak_prices = df[(df['order_date'] == peak_date) & (df['product_id'].isin(top_prods))].groupby('product_id')['unit_price'].mean()
        april_prices = df[april_mask & (df['product_id'].isin(top_prods))].groupby('product_id')['unit_price'].mean()
        
        # Calculate Price Drop
        merged = pd.concat([peak_prices, april_prices], axis=1, keys=['Peak', 'April']).dropna()
        if not merged.empty:
            merged['Drop'] = (merged['Peak'] / merged['April']) - 1
            avg_drop = merged['Drop'].mean()
            results.append({'Year': yr, 'Avg_Price_Drop': avg_drop})
            
    print("=== HOLIDAY PRICE DROP AUDIT (Top 100 Products) ===")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    audit_price_volatility()
