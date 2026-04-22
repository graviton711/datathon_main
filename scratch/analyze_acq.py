import pandas as pd
from src.config import Config

def analyze_acquisitions():
    customers = pd.read_parquet('data/processed/customers.parquet')
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    
    # Monthly Acquisitions
    customers['year'] = customers['signup_date'].dt.year
    customers['month'] = customers['signup_date'].dt.month
    
    monthly_acq = customers.groupby(['year', 'month']).size().reset_index(name='new_customers')
    monthly_acq = monthly_acq.sort_values(['year', 'month'])
    
    print("=== NEW CUSTOMER ACQUISITION TREND ===")
    print(monthly_acq.tail(24))
    
    # Compare 2022 vs 2021
    acq_2021 = monthly_acq[monthly_acq['year'] == 2021]['new_customers'].sum()
    acq_2022 = monthly_acq[monthly_acq['year'] == 2022]['new_customers'].sum()
    
    print(f"\nTotal Acquisitions 2021: {acq_2021:,}")
    print(f"Total Acquisitions 2022: {acq_2022:,}")
    print(f"Acquisition Growth: {acq_2022/acq_2021:.2f}x")

if __name__ == "__main__":
    analyze_acquisitions()
