import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def audit_viral_discounts():
    # 1. Load Data
    sales = pd.read_parquet('data/processed/sales.parquet')
    promos = pd.read_parquet('data/processed/promotions.parquet')
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    promos['start_date'] = pd.to_datetime(promos['start_date'])
    promos['end_date'] = pd.to_datetime(promos['end_date'])
    
    # 2. Daily Discount Depth (Average discount on active promos)
    # We create a daily discount series
    dates = sales['Date'].unique()
    daily_discount = []
    for d in dates:
        active = promos[(promos['start_date'] <= d) & (promos['end_date'] >= d)]
        if active.empty:
            daily_discount.append(0)
        else:
            daily_discount.append(active['discount_value'].mean())
            
    df = pd.DataFrame({'Date': dates, 'Discount': daily_discount})
    df = df.merge(sales[['Date', 'Revenue']], on='Date')
    
    # 3. Calculate Lift vs 30-day moving average
    df['Rolling_Avg'] = df['Revenue'].rolling(30, center=True).mean()
    df['Lift'] = df['Revenue'] / (df['Rolling_Avg'] + 1e-6)
    
    # 4. Bin the Discounts and check Lifts
    df['Discount_Bin'] = pd.cut(df['Discount'], bins=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], 
                               labels=['0-5%', '5-10%', '10-20%', '20-30%', '30-50%', '50%+'])
    
    print("=== DISCOUNT VIRALITY AUDIT (Non-Linear Impact) ===")
    analysis = df.groupby('Discount_Bin', observed=True)['Lift'].agg(['mean', 'median', 'max', 'count'])
    print(analysis)

if __name__ == "__main__":
    audit_viral_discounts()
