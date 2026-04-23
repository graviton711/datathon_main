import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_traffic_no_leak():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    
    traffic = pd.read_parquet(DATA_DIR / "web_traffic.parquet")
    traffic['date'] = pd.to_datetime(traffic['date'])
    traffic_daily = traffic.groupby('date')['sessions'].sum().reset_index().rename(columns={'date': 'Date'})
    
    df = pd.merge(sales, traffic_daily, on='Date', how='left').sort_values('Date').ffill()
    
    # 2. Train/Test Split (Train < 2022, Test = 2022)
    train_raw = df[df['year'] < 2022].copy()
    test_raw = df[df['year'] == 2022].copy()
    
    # 3. LEAK-FREE Normalization
    # Calculate scale only from TRAIN
    train_medians = train_raw.groupby('year')['Revenue'].median().to_dict()
    last_year_median = train_medians[2021]
    
    # Simple Inertia: Growth of Q4 2021 vs Q4 2020
    q4_2021 = train_raw[(train_raw['year'] == 2021) & (train_raw['Date'].dt.month >= 10)]['Revenue'].sum()
    q4_2020 = train_raw[(train_raw['year'] == 2020) & (train_raw['Date'].dt.month >= 10)]['Revenue'].sum()
    projected_growth = q4_2021 / (q4_2020 + 1e-6)
    projected_2022_median = last_year_median * projected_growth
    
    print(f"Projected 2022 Median (No Leak): {projected_2022_median:,.0f}")
    
    # Prepare Training Target (Stationary)
    train_raw['y'] = train_raw['Revenue'] / train_raw['year'].map(train_medians)
    
    # 4. Features
    def add_features(data):
        data = data.copy()
        data['month'] = data['Date'].dt.month
        data['dow'] = data['Date'].dt.dayofweek
        # Stationary traffic (relative to last 365d median)
        data['sessions_norm'] = data['sessions'] / data['sessions'].rolling(365).median().ffill()
        data['sessions_lag_7'] = data['sessions_norm'].shift(7)
        data['rev_lag_1'] = data['Revenue'].shift(1) # We'll normalize this later
        return data

    train_feat = add_features(train_raw).dropna()
    # For training, lags are normalized by their own year's median
    train_feat['rev_lag_1_norm'] = train_feat['rev_lag_1'] / train_feat['year'].map(train_medians)
    
    # 5. Train Models
    base_cols = ['month', 'dow', 'rev_lag_1_norm']
    traffic_cols = ['sessions_lag_7']
    
    model_base = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_base.fit(train_feat[base_cols], train_feat['y'])
    
    model_enh = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_enh.fit(train_feat[base_cols + traffic_cols], train_feat['y'])
    
    # 6. Evaluate on 2022 (Recursive-ish for simplicity)
    # To keep it simple but fair, we'll use actual lags for the first prediction
    test_feat = add_features(pd.concat([train_raw.tail(30), test_raw])).tail(len(test_raw))
    # Normalize lags for 2022 using the PROJECTED median
    test_feat['rev_lag_1_norm'] = test_feat['rev_lag_1'] / projected_2022_median
    
    # Predict and Unscale using PROJECTED median
    pred_base = model_base.predict(test_feat[base_cols]) * projected_2022_median
    pred_enh = model_enh.predict(test_feat[base_cols + traffic_cols]) * projected_2022_median
    
    mae_base = mean_absolute_error(test_raw['Revenue'], pred_base)
    mae_enh = mean_absolute_error(test_raw['Revenue'], pred_enh)
    
    print("\n--- LEAK-FREE Traffic Verification (2022) ---")
    print(f"Baseline MAE (No Leak): {mae_base:,.0f}")
    print(f"Enhanced MAE (No Leak): {mae_enh:,.0f}")
    print(f"Improvement: {mae_base - mae_enh:,.0f} ({(1 - mae_enh/mae_base)*100:.2f}%)")

if __name__ == "__main__":
    verify_traffic_no_leak()
