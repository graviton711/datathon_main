import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
SALES_FILE = DATA_DIR / "sales.parquet"

def analyze_market_heat():
    print("Analyzing Market Heat Momentum...")
    df = pd.read_parquet(SALES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate YoY Growth for different windows
    # Window 1: 30-day (Recent Heat)
    # Window 2: 365-day (Annual Baseline)
    
    # Helper to calculate YoY
    def get_yoy_ma(window):
        ma = df['Revenue'].rolling(window).mean()
        ma_prev = df['Revenue'].shift(365).rolling(window).mean()
        return ma / (ma_prev + 1e-6)
    
    df['yoy_30d'] = get_yoy_ma(30)
    df['yoy_365d'] = get_yoy_ma(365)
    
    # Market Heat Index: How much is recent growth exceeding annual baseline?
    df['market_heat'] = df['yoy_30d'] / (df['yoy_365d'] + 1e-6)
    
    # Check if market_heat predicts NEXT month's revenue shift
    df['next_30d_rev'] = df['Revenue'].shift(-30).rolling(30).mean()
    df['prev_30d_rev'] = df['Revenue'].rolling(30).mean()
    df['rev_shift'] = df['next_30d_rev'] / (df['prev_30d_rev'] + 1e-6)
    
    corr = df[['market_heat', 'rev_shift']].corr().iloc[0, 1]
    print(f"\n--- Market Heat Correlation ---")
    print(f"Correlation (Heat today vs Rev Shift in 30 days): {corr:.4f}")
    
    # Plot
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['market_heat'], label='Market Heat Index', color='orange')
    plt.axhline(y=1.0, color='red', linestyle='--')
    plt.title('Market Heat Index (Short-term YoY / Long-term YoY)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['Date'], df['Revenue'], label='Revenue', color='blue')
    plt.title('Revenue Trend')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('scratch/market_heat_analysis.png')
    print("\nAnalysis saved to scratch/market_heat_analysis.png")

if __name__ == "__main__":
    analyze_market_heat()
