import pandas as pd
import numpy as np

def analyze_yoy():
    # Load sales data
    df = pd.read_csv('data/raw/sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    
    # 1. Annual Aggregation
    annual = df.groupby('Year')['Revenue'].agg(['sum', 'mean', 'count']).reset_index()
    annual['YoY_Growth'] = annual['sum'].pct_change() * 100
    
    # 2. Q4 Comparison (Momentum check)
    df['Month'] = df['Date'].dt.month
    q4_data = df[df['Month'] >= 10].groupby('Year')['Revenue'].sum().reset_index()
    q4_data['Q4_YoY_Growth'] = q4_data['Revenue'].pct_change() * 100
    
    print("\n" + "="*60)
    print("HISTORICAL GROWTH ANALYSIS (2012 - 2022)")
    print("-" * 60)
    print(annual[['Year', 'sum', 'YoY_Growth']].to_string(index=False, formatters={'sum': '{:,.0f}'.format, 'YoY_Growth': '{:+.1f}%'.format}))
    print("-" * 60)
    
    print("\n" + "="*60)
    print("Q4 MOMENTUM ANALYSIS (Last quarter of each year)")
    print("-" * 60)
    print(q4_data[['Year', 'Revenue', 'Q4_YoY_Growth']].tail(5).to_string(index=False, formatters={'Revenue': '{:,.0f}'.format, 'Q4_YoY_Growth': '{:+.1f}%'.format}))
    print("="*60 + "\n")
    
    # Conclusion logic
    latest_yoy = annual['YoY_Growth'].iloc[-1]
    latest_q4_yoy = q4_data['Q4_YoY_Growth'].iloc[-1]
    
    if latest_q4_yoy > 0:
        print(f"INSIGHT: The business ended 2022 with {latest_q4_yoy:.1f}% Q4 GROWTH.")
        print("Hypothesis: Predicting a flat 2023 based on 2022 median will UNDER-forecast the leaderboard.")
    else:
        print(f"INSIGHT: The business ended 2022 with {latest_q4_yoy:.1f}% Q4 DECLINE.")
        print("Hypothesis: The market scale is still shrinking.")

if __name__ == "__main__":
    analyze_yoy()
