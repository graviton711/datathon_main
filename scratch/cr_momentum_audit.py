import pandas as pd
import numpy as np

def discover_cr_signals():
    # 1. Load Data
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    orders = pd.read_parquet('data/processed/orders.parquet')
    sales = pd.read_parquet('data/processed/sales.parquet')
    
    traffic['date'] = pd.to_datetime(traffic['date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    daily_traffic = traffic.groupby('date')['sessions'].sum()
    daily_orders = orders.groupby('order_date')['order_id'].count()
    
    # Daily CR
    cr = daily_orders / (daily_traffic + 1e-6)
    
    # Analyze May 1st across years
    may1_dates = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day == 1)]['Date']
    
    results = []
    for peak_date in may1_dates:
        yr = peak_date.year
        
        # Calculate Peak Lift
        baseline = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == 5) & (sales['Date'].dt.day != 1)]['Revenue'].mean()
        lift = sales[sales['Date'] == peak_date]['Revenue'].sum() / baseline
        
        # Potential Signal: CR Momentum (April vs March)
        april_cr = cr[(cr.index.year == yr) & (cr.index.month == 4)].mean()
        march_cr = cr[(cr.index.year == yr) & (cr.index.month == 3)].mean()
        cr_mom = april_cr / (march_cr + 1e-6)
        
        results.append({
            'Year': yr,
            'Peak_Lift': lift,
            'CR_MoM': cr_mom
        })
    
    df_results = pd.DataFrame(results)
    print("=== SEARCHING FOR CR MOMENTUM AS A PREDICTOR ===")
    print(df_results)
    print("\n=== CORRELATION WITH PEAK LIFT ===")
    print(df_results.corr()['Peak_Lift'])

if __name__ == "__main__":
    discover_cr_signals()
