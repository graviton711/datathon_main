import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
ORDER_ITEMS_FILE = DATA_DIR / "order_items.parquet"
ORDERS_FILE = DATA_DIR / "orders.parquet"
PRODUCTS_FILE = DATA_DIR / "products.parquet"

def verify_signals():
    print("Loading data for signal verification...")
    orders = pd.read_parquet(ORDERS_FILE)[['order_id', 'order_date']]
    items = pd.read_parquet(ORDER_ITEMS_FILE)
    products = pd.read_parquet(PRODUCTS_FILE)[['product_id', 'category']]
    
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    items = pd.merge(items, orders, on='order_id')
    items = pd.merge(items, products, on='product_id')
    
    items['item_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
    items['month'] = items['order_date'].dt.month
    items['year'] = items['order_date'].dt.year
    
    # 1. Verify Recurring Promo Windows Lift
    # Windows: Feb(2), June(6), Aug(8), Nov(11)
    promo_months = [2, 6, 8, 11]
    
    monthly_rev = items.groupby(['year', 'month'])['item_rev'].sum().reset_index()
    monthly_rev['is_promo_month'] = monthly_rev['month'].isin(promo_months)
    
    lift_stats = monthly_rev.groupby('is_promo_month')['item_rev'].mean()
    print("\n--- Recurring Promo Window Lift ---")
    print(lift_stats)
    print(f"Average Lift: {lift_stats[True] / lift_stats[False]:.2f}x")
    
    # 2. Verify Category Contribution Stability
    cat_monthly = items.groupby(['year', 'month', 'category'])['item_rev'].sum().unstack().fillna(0)
    cat_shares = cat_monthly.div(cat_monthly.sum(axis=1), axis=0)
    
    print("\n--- Category Share Stability (Std Dev of Shares) ---")
    print(cat_shares.std())
    
    # 3. Monthly Category Profiles (Average share per month across years)
    cat_profile = cat_shares.reset_index().groupby('month').mean().drop(columns='year')
    print("\n--- Monthly Category Revenue Profile (Mean Share) ---")
    print(cat_profile)
    
    # 4. Impact of Category Shift on Total Revenue
    # Correlation between Streetwear Share and Total Revenue Efficiency
    # (Assuming we have total revenue per month)
    monthly_rev_total = monthly_rev.set_index(['year', 'month'])['item_rev']
    streetwear_share = cat_shares['Streetwear']
    
    correlation = streetwear_share.corr(monthly_rev_total)
    print(f"\nCorrelation between Streetwear Share and Total Revenue: {correlation:.4f}")
    
    # Visualization of Category Shifts
    cat_profile.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Monthly Category Revenue Mix (Historical Average)')
    plt.ylabel('Revenue Share')
    plt.savefig('scratch/category_signal_verification.png')
    print("\nVisualization saved to scratch/category_signal_verification.png")

if __name__ == "__main__":
    verify_signals()
