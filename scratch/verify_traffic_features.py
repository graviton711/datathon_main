import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_traffic_features():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    
    traffic = pd.read_parquet(DATA_DIR / "web_traffic.parquet")
    traffic['date'] = pd.to_datetime(traffic['date'])
    traffic_daily = traffic.groupby('date')['sessions'].sum().reset_index().rename(columns={'date': 'Date'})
    
    # 2. Basic Feature Engineering
    df = pd.merge(sales, traffic_daily, on='Date', how='left').fillna(method='ffill')
    
    # Normalization (Like in our pipeline)
    annual_medians = df.groupby('year')['Revenue'].median().to_dict()
    df['y'] = df['Revenue'] / df['year'].map(annual_medians)
    
    # Lag Features
    df['rev_lag_1'] = df['y'].shift(1)
    df['rev_lag_7'] = df['y'].shift(7)
    
    # Traffic Features (Lagged to avoid leakage)
    df['sessions_norm'] = df['sessions'] / df['sessions'].rolling(365).median() # Stationary traffic
    df['sessions_lag_7'] = df['sessions_norm'].shift(7)
    df['sessions_lag_30'] = df['sessions_norm'].shift(30)
    
    # Time features
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    df = df.dropna()
    
    # 3. Split: Train < 2022, Test = 2022
    train = df[df['year'] < 2022].copy()
    test = df[df['year'] == 2022].copy()
    
    base_features = ['month', 'day_of_week', 'rev_lag_1', 'rev_lag_7']
    traffic_features = ['sessions_lag_7', 'sessions_lag_30']
    
    # 4. Train Baseline
    model_base = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_base.fit(train[base_features], train['y'])
    pred_base = model_base.predict(test[base_features]) * test['year'].map(annual_medians)
    mae_base = mean_absolute_error(test['Revenue'], pred_base)
    
    # 5. Train Enhanced
    model_enh = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_enh.fit(train[base_features + traffic_features], train['y'])
    pred_enh = model_enh.predict(test[base_features + traffic_features]) * test['year'].map(annual_medians)
    mae_enh = mean_absolute_error(test['Revenue'], pred_enh)
    
    print("--- Traffic Feature Verification (Test Year: 2022) ---")
    print(f"Baseline MAE: {mae_base:,.0f}")
    print(f"Enhanced MAE (with Traffic Lags): {mae_enh:,.0f}")
    print(f"Improvement: {mae_base - mae_enh:,.0f} ({(1 - mae_enh/mae_base)*100:.2f}%)")
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': base_features + traffic_features,
        'importance': model_enh.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n--- Feature Importance ---")
    print(importances)

if __name__ == "__main__":
    verify_traffic_features()
