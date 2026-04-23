import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SALES_FILE = "data/processed/sales.parquet"
TRAFFIC_FILE = "data/processed/web_traffic.parquet"

def verify_weighting():
    # 1. Load Data
    sales = pd.read_parquet(SALES_FILE)
    traffic = pd.read_parquet(TRAFFIC_FILE)
    
    sales['Date'] = pd.to_datetime(sales['Date'])
    traffic['date'] = pd.to_datetime(traffic['date'])
    
    # 2. Aggregate Monthly
    monthly_sales = sales.groupby(sales['Date'].dt.to_period('M'))['Revenue'].sum().reset_index()
    monthly_sales.columns = ['Month', 'Revenue']
    monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()
    
    monthly_traffic = traffic.groupby(traffic['date'].dt.to_period('M'))['sessions'].sum().reset_index()
    monthly_traffic.columns = ['Month', 'Sessions']
    monthly_traffic['Month'] = monthly_traffic['Month'].dt.to_timestamp()
    
    df = pd.merge(monthly_sales, monthly_traffic, on='Month')
    
    # 3. Calculate Efficiency Ratio
    df['Efficiency_Ratio'] = df['Revenue'] / (df['Sessions'] + 1e-6)
    
    # 4. Statistical Distribution (Yearly Context to avoid historical bias)
    df['Year'] = df['Month'].dt.year
    
    yearly_stats = df.groupby('Year')['Efficiency_Ratio'].agg(['median', 'std']).reset_index()
    yearly_stats.columns = ['Year', 'Yearly_Median', 'Yearly_Std']
    
    df = pd.merge(df, yearly_stats, on='Year')
    
    # Calculate Z relative to yearly context
    df['Z_Score'] = (df['Efficiency_Ratio'] - df['Yearly_Median']).abs() / (df['Yearly_Std'] + 1e-6)
    df['Weight'] = 1.0 / (1.0 + df['Z_Score'])
    
    # 5. Output Results for Review
    print("\n--- REFINED TOP 10 ANOMALOUS MONTHS (Yearly Context) ---")
    top_anomalies = df.sort_values('Weight').head(10)
    print(top_anomalies[['Month', 'Revenue', 'Sessions', 'Efficiency_Ratio', 'Z_Score', 'Weight']])
    
    # 6. Visualization (Save to scratch)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df['Month'], df['Efficiency_Ratio'], label='Efficiency Ratio', color='blue')
    for yr in df['Year'].unique():
        yr_val = yearly_stats[yearly_stats['Year'] == yr]['Yearly_Median'].values[0]
        yr_dates = df[df['Year'] == yr]['Month']
        plt.plot(yr_dates, [yr_val]*len(yr_dates), color='red', linestyle='--')
    plt.title('Monthly Efficiency Ratio vs Yearly Medians')
    
    plt.subplot(3, 1, 2)
    plt.bar(df['Month'], df['Weight'], width=20, color='green', alpha=0.6, label='Calculated Weight')
    plt.title('Sample Weights (Relative to Yearly Median)')
    plt.ylim(0, 1.1)
    
    plt.subplot(3, 1, 3)
    plt.plot(df['Month'], df['Revenue'], label='Actual Revenue', color='black')
    plt.title('Reference: Actual Revenue Trend')
    
    plt.tight_layout()
    plt.savefig('scratch/weighting_verification.png')
    print("\nPlot saved to scratch/weighting_verification.png")

if __name__ == "__main__":
    verify_weighting()
