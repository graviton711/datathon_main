import pandas as pd
from src.config import Config

def analyze_cr_acceleration():
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    traffic['date'] = pd.to_datetime(traffic['date'])
    daily_traffic = traffic.groupby('date')['sessions'].sum().reset_index()
    
    orders = pd.read_parquet('data/processed/orders.parquet')
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    daily_orders = orders.groupby('order_date')['order_id'].count().reset_index()
    daily_orders.columns = ['date', 'orders']
    
    # Merge and calculate CR
    df = pd.merge(daily_traffic, daily_orders, on='date', how='left').fillna(0)
    df['cr'] = df['orders'] / (df['sessions'] + 1e-6)
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    monthly_cr = df.groupby(['year', 'month'])['cr'].mean().reset_index()
    monthly_cr = monthly_cr.sort_values(['year', 'month'])
    
    print("=== CONVERSION RATE (CR) TREND ===")
    print(monthly_cr.tail(24))
    
    # Compare Q4 2022 vs Q4 2021
    cr_q4_21 = monthly_cr[(monthly_cr['year'] == 2021) & (monthly_cr['month'] >= 10)]['cr'].mean()
    cr_q4_22 = monthly_cr[(monthly_cr['year'] == 2022) & (monthly_cr['month'] >= 10)]['cr'].mean()
    
    print(f"\nAvg CR Q4 2021: {cr_q4_21:.4f}")
    print(f"Avg CR Q4 2022: {cr_q4_22:.4f}")
    print(f"CR Growth (Q4 YoY): {cr_q4_22/cr_q4_21:.2f}x")

if __name__ == "__main__":
    analyze_cr_acceleration()
