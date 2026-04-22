import pandas as pd
import numpy as np

def signal_hunt():
    # 1. Load Data
    orders = pd.read_parquet('data/processed/orders.parquet')
    order_items = pd.read_parquet('data/processed/order_items.parquet')
    sales = pd.read_parquet('data/processed/sales.parquet')
    products = pd.read_parquet('data/processed/products.parquet')
    
    # 2. Daily Metrics Construction
    # A. Promo Activity
    daily_promo = order_items.groupby(order_items['order_id']).agg({
        'discount_amount': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    # B. Merge with Orders for Dates
    daily_promo = daily_promo.merge(orders[['order_id', 'order_date']], on='order_id')
    daily_promo['order_date'] = pd.to_datetime(daily_promo['order_date'])
    
    promo_stats = daily_promo.groupby('order_date').agg({
        'discount_amount': 'sum',
        'quantity': 'sum',
        'order_id': 'count'
    }).rename(columns={'order_id': 'order_count'})
    
    # C. Merge with Sales (Target)
    sales['Date'] = pd.to_datetime(sales['Date'])
    df = sales.merge(promo_stats, left_on='Date', right_index=True, how='left').fillna(0)
    
    # D. Calculate Derived Metrics
    df['AOV'] = df['Revenue'] / (df['order_count'] + 1e-6)
    df['discount_ratio'] = df['discount_amount'] / (df['Revenue'] + df['discount_amount'] + 1e-6)
    
    # 3. High Correlation Search
    print("=== SEARCHING FOR HIGH CORRELATION SIGNALS ===")
    corrs = df.corr()['Revenue'].sort_values(ascending=False)
    print(corrs)
    
    # 4. Analyze the Peaks (Top 5% Revenue days)
    threshold = df['Revenue'].quantile(0.95)
    peaks = df[df['Revenue'] >= threshold]
    normals = df[df['Revenue'] < threshold]
    
    print(f"\n=== PEAK vs NORMAL DAYS ANALYSIS (Threshold: {threshold:,.0f}) ===")
    stats = pd.DataFrame({
        'Metric': ['AOV', 'discount_ratio', 'quantity_per_order'],
        'Normal_Avg': [normals['AOV'].mean(), normals['discount_ratio'].mean(), (normals['quantity']/normals['order_count']).mean()],
        'Peak_Avg': [peaks['AOV'].mean(), peaks['discount_ratio'].mean(), (peaks['quantity']/peaks['order_count']).mean()]
    })
    print(stats)

if __name__ == "__main__":
    signal_hunt()
