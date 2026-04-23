import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
DATA_DIR = Path("data/processed")
ORDERS_FILE = DATA_DIR / "orders.parquet"
TRAFFIC_FILE = DATA_DIR / "web_traffic.parquet"

def analyze_promo():
    print("Loading data...")
    orders = pd.read_parquet(ORDERS_FILE)[['order_id', 'order_date']]
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    traffic = pd.read_parquet(TRAFFIC_FILE)
    
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    traffic['date'] = pd.to_datetime(traffic['date'])
    
    # 1. Join Items with Orders for Date
    items = pd.merge(items, orders, on='order_id', how='inner')
    
    # 2. Calculate Daily Promo Metrics
    # Item-level revenue = qty * price - discount
    items['item_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
    items['is_promo'] = items['promo_id'].notna()
    
    daily = items.groupby('order_date').agg({
        'item_rev': 'sum',
        'discount_amount': 'sum',
        'is_promo': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    daily.columns = ['date', 'revenue', 'total_discount', 'promo_item_count', 'total_orders']
    
    # Merge with traffic
    df = pd.merge(daily, traffic, on='date', how='inner')
    
    # 3. Calculate Promo Intensity Metrics
    df['promo_intensity'] = df['promo_item_count'] / (df['total_orders'] + 1e-6)
    df['discount_depth'] = df['total_discount'] / (df['revenue'] + df['total_discount'] + 1e-6)
    df['efficiency'] = df['revenue'] / (df['sessions'] + 1e-6)
    
    # 4. Monthly Resampling for Trend Analysis
    df['month_dt'] = df['date'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('month_dt').agg({
        'promo_intensity': 'mean',
        'discount_depth': 'mean',
        'efficiency': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    # 5. Correlations
    corr_matrix = monthly[['promo_intensity', 'discount_depth', 'efficiency']].corr()
    print("\n--- Correlation Matrix (Monthly) ---")
    print(corr_matrix)
    
    # 6. Yearly Stats
    monthly['year'] = monthly['month_dt'].dt.year
    yearly = monthly.groupby('year').agg({
        'promo_intensity': 'mean',
        'discount_depth': 'mean',
        'efficiency': 'mean'
    }).reset_index()
    
    print("\n--- Yearly Promo Evolution ---")
    print(yearly)
    
    # 7. Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(monthly['month_dt'], monthly['promo_intensity'], color='orange', label='Promo Intensity (Items/Order)')
    plt.title('Trend of Promo Intensity over Time')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(monthly['month_dt'], monthly['efficiency'], color='blue', label='Revenue Efficiency (Rev/Session)')
    plt.title('Trend of Revenue Efficiency over Time')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    sns.regplot(data=monthly, x='promo_intensity', y='efficiency', scatter_kws={'alpha':0.5})
    plt.title('Correlation: Promo Intensity vs Efficiency')
    
    plt.tight_layout()
    plt.savefig('scratch/promo_analysis.png')
    print("\nAnalysis saved to scratch/promo_analysis.png")

if __name__ == "__main__":
    analyze_promo()
