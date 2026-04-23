import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
DATA_DIR = Path("data/processed")
SALES_FILE = DATA_DIR / "sales.parquet"
TRAFFIC_FILE = DATA_DIR / "web_traffic.parquet"

def analyze_traffic_signal():
    print("Loading data...")
    sales = pd.read_parquet(SALES_FILE)
    traffic = pd.read_parquet(TRAFFIC_FILE)
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    traffic['date'] = pd.to_datetime(traffic['date'])
    
    # 1. Pivot Traffic by Source
    traffic_pivot = traffic.pivot_table(
        index='date', 
        columns='traffic_source', 
        values='sessions', 
        aggfunc='sum'
    ).fillna(0)
    
    # Calculate Total Sessions and Share
    traffic_pivot['total_sessions'] = traffic_pivot.sum(axis=1)
    sources = [c for c in traffic_pivot.columns if c != 'total_sessions']
    for s in sources:
        traffic_pivot[f'share_{s}'] = traffic_pivot[s] / traffic_pivot['total_sessions']
    
    # 2. Merge with Sales
    df = pd.merge(sales, traffic_pivot, left_on='Date', right_index=True, how='inner')
    df['efficiency'] = df['Revenue'] / (df['total_sessions'] + 1e-6)
    
    # 3. Monthly Trend
    df['month_dt'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('month_dt').agg({
        'efficiency': 'mean',
        **{f'share_{s}': 'mean' for s in sources}
    }).reset_index()
    
    # 4. Correlation with Efficiency
    share_cols = [f'share_{s}' for s in sources]
    corr_matrix = monthly[['efficiency'] + share_cols].corr()
    print("\n--- Correlation: Traffic Source Share vs Efficiency ---")
    print(corr_matrix['efficiency'].sort_values(ascending=False))
    
    # 5. Visualization
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 1, 1)
    for s in sources:
        plt.plot(monthly['month_dt'], monthly[f'share_{s}'], label=s)
    plt.title('Traffic Source Mix over Time')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(monthly['month_dt'], monthly['efficiency'], color='black', linewidth=2, label='Efficiency')
    plt.title('Revenue Efficiency Trend')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('scratch/traffic_source_analysis.png')
    print("\nAnalysis saved to scratch/traffic_source_analysis.png")

if __name__ == "__main__":
    analyze_traffic_signal()
