import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_signals():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    traffic = pd.read_parquet(DATA_DIR / "web_traffic.parquet")
    traffic['date'] = pd.to_datetime(traffic['date'])
    
    reviews = pd.read_parquet(DATA_DIR / "reviews.parquet")
    reviews['review_date'] = pd.to_datetime(reviews['review_date'])
    
    # 2. Daily Processing
    traffic_daily = traffic.groupby('date')['sessions'].sum().reset_index().rename(columns={'date': 'Date'})
    reviews_daily = reviews.groupby('review_date')['rating'].mean().reset_index().rename(columns={'review_date': 'Date'})
    reviews_daily['rating_30d'] = reviews_daily['rating'].rolling(30).mean()
    
    # 3. Aggregations
    df = pd.merge(sales, traffic_daily, on='Date', how='left')
    df = pd.merge(df, reviews_daily[['Date', 'rating_30d']], on='Date', how='left')
    df = df.sort_values('Date').ffill()
    
    # Calculate 30-day moving averages to remove noise
    df['rev_30d'] = df['Revenue'].rolling(30).mean()
    df['sessions_30d'] = df['sessions'].rolling(30).mean()
    
    # Correlation of 30-day averages
    print("--- 30-day Rolling Correlation ---")
    corr_matrix = df[['rev_30d', 'sessions_30d', 'rating_30d']].corr()
    print(corr_matrix)
    
    # Lag Correlation: Does rating_30d at t predict rev_30d at t+X?
    print("\n--- Lag Correlation (Signal predicts Future Revenue) ---")
    for lag in [30, 60, 90, 180]:
        c = df['rev_30d'].corr(df['rating_30d'].shift(lag))
        print(f"Rating (lag {lag} days) vs Revenue: {c:.3f}")

    for lag in [7, 14, 30]:
        c = df['rev_30d'].corr(df['sessions_30d'].shift(lag))
        print(f"Sessions (lag {lag} days) vs Revenue: {c:.3f}")

if __name__ == "__main__":
    analyze_signals()
