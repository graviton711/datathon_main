import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def verify_yoy_power():
    print("Verifying YoY Signal Power (1-Step Ahead Test)...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Prepare data with 364 lag
    df = sales.copy().sort_values('Date')
    df['rev_lag_1'] = df['Revenue'].shift(1)
    df['rev_lag_7'] = df['Revenue'].shift(7)
    df['rev_roll_7'] = df['Revenue'].shift(1).rolling(7).mean()
    df['rev_lag_364'] = df['Revenue'].shift(364)
    
    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    
    # Features
    base_features = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7']
    exp_features = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7', 'rev_lag_364']
    
    # Simple Train/Test Split (Train until 2021, Test 2022)
    train = df[df['Date'].dt.year < 2022].copy()
    test = df[df['Date'].dt.year == 2022].copy()
    
    y_train = train['Revenue']
    y_test = test['Revenue']
    
    # Model Base
    model_base = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model_base.fit(train[base_features], y_train)
    pred_base = model_base.predict(test[base_features])
    mae_base = np.abs(pred_base - y_test).mean()
    
    # Model Exp
    model_exp = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model_exp.fit(train[exp_features], y_train)
    pred_exp = model_exp.predict(test[exp_features])
    mae_exp = np.abs(pred_exp - y_test).mean()
    
    print(f"\n--- 1-Step Ahead Results (2022) ---")
    print(f"MAE Base (1, 7, roll7)    : {mae_base:,.0f}")
    print(f"MAE Exp (+ lag 364)        : {mae_exp:,.0f}")
    print(f"Improvement                : {mae_base - mae_exp:,.0f} ({(1 - mae_exp/mae_base)*100:.1f}%)")

if __name__ == "__main__":
    verify_yoy_power()
