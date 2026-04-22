import pandas as pd
import numpy as np

def discover_patterns():
    # 1. Load all processed data
    sales = pd.read_parquet('data/processed/sales.parquet')
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    customers = pd.read_parquet('data/processed/customers.parquet')
    
    # 2. Daily Aggregation
    sales['Date'] = pd.to_datetime(sales['Date'])
    traffic['date'] = pd.to_datetime(traffic['date'])
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    
    daily_sales = sales.groupby('Date')['Revenue'].sum().reset_index()
    daily_traffic = traffic.groupby('date')[['sessions', 'unique_visitors']].sum().reset_index()
    daily_signups = customers.groupby('signup_date').size().reset_index(name='new_signups')
    
    # 3. Merge all signals
    df = daily_sales.merge(daily_traffic, left_on='Date', right_on='date', how='left')
    df = df.merge(daily_signups, left_on='Date', right_on='signup_date', how='left').fillna(0)
    
    # 4. Correlation Analysis
    print("=== CORRELATION WITH DAILY REVENUE ===")
    corr = df[['Revenue', 'sessions', 'unique_visitors', 'new_signups']].corr()['Revenue']
    print(corr)
    
    # 5. Look for "Golden Rules"
    # Rule 1: Revenue per Session (Efficiency)
    df['rev_per_session'] = df['Revenue'] / (df['sessions'] + 1e-6)
    print("\n=== REVENUE PER SESSION (EFFICIENCY) OVER YEARS ===")
    df['year'] = df['Date'].dt.year
    print(df.groupby('year')['rev_per_session'].mean())
    
    # Rule 2: Signup to Revenue Ratio
    df['rev_per_signup'] = df['Revenue'] / (df['new_signups'] + 1e-6)
    print("\n=== REVENUE PER NEW SIGNUP OVER YEARS ===")
    print(df.groupby('year')['rev_per_signup'].mean())

if __name__ == "__main__":
    discover_patterns()
