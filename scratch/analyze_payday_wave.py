import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
SALES_FILE = DATA_DIR / "sales.parquet"

def analyze_payday_wave():
    print("Analyzing Payday Wave Intensity...")
    df = pd.read_parquet(SALES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.day
    
    # Calculate Mean Revenue per Day of Month
    day_profile = df.groupby('day')['Revenue'].mean().reset_index()
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(day_profile['day'], day_profile['Revenue'], marker='o', color='purple')
    plt.axvspan(25, 31, alpha=0.2, color='green', label='Payday Window')
    plt.axvspan(1, 5, alpha=0.2, color='green')
    plt.title('Average Revenue by Day of Month (Payday Cycle)')
    plt.xlabel('Day of Month')
    plt.ylabel('Average Revenue')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('scratch/payday_wave_analysis.png')
    
    # Check if certain months have SHARPER payday waves
    df['month'] = df['Date'].dt.month
    monthly_day_profile = df.groupby(['month', 'day'])['Revenue'].mean().unstack()
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(monthly_day_profile, cmap='YlGnBu')
    plt.title('Heatmap: Revenue by Month and Day (Identifying Payday Sharpness)')
    plt.savefig('scratch/payday_heatmap.png')
    
    print("\nAnalysis saved to scratch/payday_heatmap.png")

if __name__ == "__main__":
    import seaborn as sns
    analyze_payday_wave()
