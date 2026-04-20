import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def check_predictability(name, df, date_col, val_col, freq=365):
    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.groupby(date_col)[val_col].mean().resample('D').mean().ffill()
    
    if len(daily) < freq * 2:
        return f"{name}: Insufficient data for annual decomposition"
    
    # Decomposition
    res = seasonal_decompose(daily, model='additive', period=freq)
    
    # Calculate Strength of Seasonality: 1 - Var(Resid) / Var(Detrended)
    detrended = daily - res.trend.fillna(0)
    resid = res.resid.fillna(0)
    
    # Simple metric: how much of the variance is explained by the annual cycle?
    seasonality_strength = 1 - (np.var(resid) / (np.var(detrended) + 1e-6))
    
    return {
        'name': name,
        'seasonality_strength': seasonality_strength,
        'mean': daily.mean(),
        'std': daily.std()
    }

def run_predictability_audit():
    # 1. Web Traffic
    traffic_df = pd.read_csv('data/raw/web_traffic.csv')
    res_traffic = check_predictability('Web Traffic', traffic_df, 'date', 'sessions')
    
    # 2. Sentiment
    reviews_df = pd.read_csv('data/raw/reviews.csv')
    res_sentiment = check_predictability('Sentiment', reviews_df, 'review_date', 'rating')
    
    # 3. Inventory
    inventory_df = pd.read_csv('data/raw/inventory.csv')
    # Inventory is monthly, period=12
    res_inventory = check_predictability('Inventory', inventory_df, 'snapshot_date', 'stock_on_hand', freq=12)

    print("\n" + "="*50)
    print("META-MODEL PREDICTABILITY AUDIT")
    print("-" * 50)
    for r in [res_traffic, res_sentiment, res_inventory]:
        if isinstance(r, dict):
            print(f"{r['name']}:")
            print(f"  Seasonality Strength: {r['seasonality_strength']:.2f}")
            print(f"  Volatility (Std/Mean): {r['std']/r['mean']:.2f}")
        else:
            print(r)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_predictability_audit()
