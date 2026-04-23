import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_cohorts():
    # 1. Load Data
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date', 'customer_id']]
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")[['customer_id', 'signup_date']]
    
    # 2. Join and Define Cohorts (by Signup Year)
    df = pd.merge(orders, customers, on='customer_id', how='left')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['order_year'] = df['order_date'].dt.year
    df['cohort_year'] = df['signup_date'].dt.year
    
    # 3. Calculate Revenue Proxy (Order Count) per Cohort per Year
    cohort_matrix = df.groupby(['order_year', 'cohort_year']).size().unstack(fill_value=0)
    
    print("--- Cohort Order Matrix (Rows: Order Year, Cols: Signup Year) ---")
    print(cohort_matrix.tail(10))
    
    # 4. Calculate Retention Strength
    # Defined as: Orders from "Old" customers (signed up > 1 year ago) / Total Orders
    cohort_matrix_norm = cohort_matrix.div(cohort_matrix.sum(axis=1), axis=0)
    
    # "Loyalty Ratio": Share of orders from customers who signed up at least 2 years before the order year
    loyalty_ratios = {}
    for yr in cohort_matrix.index:
        old_orders = cohort_matrix.loc[yr, [c for c in cohort_matrix.columns if c <= yr - 2]].sum()
        total_orders = cohort_matrix.loc[yr].sum()
        loyalty_ratios[yr] = old_orders / (total_orders + 1e-6)
        
    loyalty_df = pd.Series(loyalty_ratios, name='loyalty_ratio')
    print("\n--- Loyalty Ratio Over Years (Share of orders from 2yr+ veterans) ---")
    print(loyalty_df)
    
    # 5. Check Correlation with Next Year Growth
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    annual_medians = sales.groupby('year')['Revenue'].median()
    growth = annual_medians.shift(-1) / annual_medians - 1.0
    
    research_df = pd.concat([loyalty_df, growth.rename('next_year_growth')], axis=1).dropna()
    print("\n--- Correlation: Loyalty Ratio vs Next Year Growth ---")
    print(research_df.corr())
    
    # 6. Plotting
    plt.figure(figsize=(12, 6))
    cohort_matrix_norm.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis')
    plt.title("Revenue Mix by Customer Signup Cohort")
    plt.ylabel("Share of Orders")
    plt.legend(title="Signup Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("e:/VSCODE_WORKSPACE/NewDatathon/data/plots/cohort_analysis.png")
    
if __name__ == "__main__":
    analyze_cohorts()
