import pandas as pd
import numpy as np

def final_formula_audit():
    # 1. Load Data
    sales = pd.read_parquet('data/processed/sales.parquet')
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    orders = pd.read_parquet('data/processed/orders.parquet')
    
    # 2. Daily Merge
    sales['Date'] = pd.to_datetime(sales['Date'])
    traffic['date'] = pd.to_datetime(traffic['date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    daily_traffic = traffic.groupby('date')['sessions'].sum().reset_index()
    daily_orders = orders.groupby('order_date')['order_id'].count().reset_index(name='order_count')
    
    df = daily_orders.merge(daily_traffic, left_on='order_date', right_on='date', how='left')
    df = df.merge(sales[['Date', 'Revenue']], left_on='order_date', right_on='Date', how='left').fillna(0)
    
    # 3. Calculate Daily Conversion Rate (CR)
    df['CR'] = df['order_count'] / (df['sessions'] + 1e-6)
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    
    print("=== CONVERSION RATE (CR) TREND OVER YEARS ===")
    yearly_cr = df.groupby('year')['CR'].mean()
    print(yearly_cr)
    
    # 4. Calculate Customer Retention Signal
    # Ratio of orders from "Old" customers vs "New" customers
    customers = pd.read_parquet('data/processed/customers.parquet')
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    
    order_cust = orders.merge(customers[['customer_id', 'signup_date']], on='customer_id')
    order_cust['order_date'] = pd.to_datetime(order_cust['order_date'])
    order_cust['is_new_customer'] = (order_cust['order_date'].dt.date == order_cust['signup_date'].dt.date).astype(int)
    
    retention_stats = order_cust.groupby(order_cust['order_date'].dt.year).agg({
        'order_id': 'count',
        'is_new_customer': 'sum'
    })
    retention_stats['returning_orders'] = retention_stats['order_id'] - retention_stats['is_new_customer']
    retention_stats['retention_ratio'] = retention_stats['returning_orders'] / retention_stats['order_id']
    
    print("\n=== CUSTOMER RETENTION TREND (RETURNING ORDERS) ===")
    print(retention_stats[['retention_ratio']])
    
    # 5. Monthly Decomposition (2022 vs 2021)
    df_21_22 = df[df['year'].isin([2021, 2022])].copy()
    
    # Calculate AOV
    df_21_22['AOV'] = df_21_22['Revenue'] / (df_21_22['order_count'] + 1e-6)
    
    monthly_stats = df_21_22.groupby(['year', 'month']).agg({
        'sessions': 'mean',
        'CR': 'mean',
        'AOV': 'mean',
        'Revenue': 'sum'
    }).reset_index()
    
    # Pivot to compare 2022 vs 2021
    pivot = monthly_stats.pivot(index='month', columns='year', values=['sessions', 'CR', 'AOV', 'Revenue'])
    
    growth_factors = pd.DataFrame(index=pivot.index)
    growth_factors['Traffic_Lift'] = pivot['sessions'][2022] / pivot['sessions'][2021]
    growth_factors['CR_Lift'] = pivot['CR'][2022] / pivot['CR'][2021]
    growth_factors['AOV_Lift'] = pivot['AOV'][2022] / pivot['AOV'][2021]
    growth_factors['Total_Rev_Growth'] = pivot['Revenue'][2022] / pivot['Revenue'][2021]
    growth_factors['Formula_Check'] = growth_factors['Traffic_Lift'] * growth_factors['CR_Lift'] * growth_factors['AOV_Lift']
    
    print("\n=== MONTHLY GROWTH DECOMPOSITION (2022 vs 2021) ===")
    print(growth_factors.round(3))
    
    print("\n=== GLOBAL STABILITY CHECK ===")
    print(f"Median Formula Check: {growth_factors['Formula_Check'].median():.3f}")
    print(f"Median Actual Rev Growth: {growth_factors['Total_Rev_Growth'].median():.3f}")

if __name__ == "__main__":
    final_formula_audit()
