import pandas as pd
import numpy as np

def discover_intensity_signals():
    # 1. Load Data
    sales = pd.read_parquet('data/processed/sales.parquet')
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    customers = pd.read_parquet('data/processed/customers.parquet')
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    traffic['date'] = pd.to_datetime(traffic['date'])
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    
    # 2. Identify all "May 1st" events
    may1_events = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day == 1)].copy()
    
    results = []
    for _, event in may1_events.iterrows():
        peak_date = event['Date']
        yr = peak_date.year
        
        # Calculate Peak Lift
        baseline = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == 5) & (sales['Date'].dt.day != 1)]['Revenue'].mean()
        lift = event['Revenue'] / baseline
        
        # Potential Signal 1: Traffic Growth in April
        april_traffic = traffic[(traffic['date'].dt.year == yr) & (traffic['date'].dt.month == 4)]['sessions'].mean()
        march_traffic = traffic[(traffic['date'].dt.year == yr) & (traffic['date'].dt.month == 3)]['sessions'].mean()
        traffic_mom = april_traffic / march_traffic if march_traffic > 0 else 1.0
        
        # Potential Signal 2: New Signups in Q1 (Jan-Mar)
        q1_signups = customers[(customers['signup_date'].dt.year == yr) & (customers['signup_date'].dt.month <= 3)].shape[0]
        
        # Potential Signal 3: Revenue Trend in April (30 days before)
        april_rev = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == 4)]['Revenue'].sum()
        march_rev = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == 3)]['Revenue'].sum()
        rev_mom = april_rev / march_rev if march_rev > 0 else 1.0
        
        results.append({
            'Year': yr,
            'Peak_Lift': lift,
            'Traffic_MoM': traffic_mom,
            'Q1_Signups': q1_signups,
            'Rev_MoM': rev_mom
        })
    
    df_results = pd.DataFrame(results)
    print("=== SEARCHING FOR LEADING INDICATORS OF HOLIDAY INTENSITY ===")
    print(df_results)
    
    print("\n=== CORRELATION WITH PEAK LIFT ===")
    print(df_results.corr()['Peak_Lift'])

if __name__ == "__main__":
    discover_intensity_signals()
