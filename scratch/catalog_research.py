import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_catalog_growth():
    # 1. Load Data
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")[['order_id', 'product_id']]
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date']]
    
    df = pd.merge(items, orders, on='order_id')
    df['Date'] = pd.to_datetime(df['order_date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    
    # 2. Count Active Products per Year/Month
    # We define "Active Products" as unique product_ids sold in that period
    catalog_yearly = df.groupby('year')['product_id'].nunique()
    
    print("--- Active Product Count Over Years ---")
    print(catalog_yearly)
    
    # 3. Calculate Catalog Growth
    catalog_growth = catalog_yearly.pct_change()
    
    # 4. Correlate with Revenue Growth
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['year'] = pd.to_datetime(sales['Date']).dt.year
    annual_medians = sales.groupby('year')['Revenue'].median()
    rev_growth = annual_medians.pct_change()
    
    research_df = pd.concat([catalog_growth.rename('catalog_growth'), 
                             rev_growth.rename('revenue_growth')], axis=1).dropna()
    
    print("\n--- Correlation: Catalog Growth vs Revenue Growth ---")
    print(research_df.corr())
    
    # 5. Check Q4 Catalog as Predictor for Next Year
    q4_catalog = df[df['month'] >= 10].groupby('year')['product_id'].nunique()
    q4_catalog_growth = q4_catalog.pct_change()
    
    # Next year growth
    next_yr_rev_growth = rev_growth.shift(-1)
    
    predictor_df = pd.concat([q4_catalog_growth.rename('q4_catalog_momentum'), 
                              next_yr_rev_growth.rename('next_year_rev_growth')], axis=1).dropna()
    
    print("\n--- Predictor Check: Q4 Catalog Momentum vs Next Year Revenue Growth ---")
    print(predictor_df)
    print("\nCorrelation:", predictor_df.corr().iloc[0, 1])

if __name__ == "__main__":
    analyze_catalog_growth()
