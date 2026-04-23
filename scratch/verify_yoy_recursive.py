import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def verify_yoy_recursive():
    print("Starting Recursive Stress Test for YoY Signal (rev_lag_364)...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Use Fold 3: Train until 2021, Test 2022
    train_df = sales[sales['Date'].dt.year < 2022].copy()
    test_df = sales[sales['Date'].dt.year == 2022].copy()
    
    # 1. Baseline
    pipe_base = ForecastingPipeline()
    pipe_base.fit(train_df)
    forecast_base = pipe_base.predict(test_df[['Date']])
    mae_base = np.abs(forecast_base['Revenue'].values - test_df['Revenue'].values).mean()
    
    # 2. Experimental (The Hard Part: Integrating into the loop)
    # To avoid changing the source code, we manually simulate the recursive loop here
    # mimicking the logic in pipeline.py
    print("Training Experimental Model...")
    
    # Prepare training data with lag 364
    train_ext = train_df.copy().sort_values('Date')
    train_ext['rev_lag_1'] = train_ext['Revenue'].shift(1)
    train_ext['rev_lag_7'] = train_ext['Revenue'].shift(7)
    train_ext['rev_roll_7'] = train_ext['Revenue'].shift(1).rolling(7).mean()
    train_ext['rev_lag_364'] = train_ext['Revenue'].shift(364)
    train_ext = train_ext.dropna().reset_index(drop=True)
    
    # We use a simplified model for the stress test (mimicking the pipeline's LGBM)
    features = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7', 'rev_lag_364']
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(train_ext[features], train_ext['Revenue'])
    
    # Recursive Loop
    print("Running Recursive Loop (540 steps simulation)...")
    history = list(train_ext['Revenue'].values) # Buffer
    preds = []
    
    # We'll just predict for the 2022 period (365 days)
    for i in range(len(test_df)):
        # Extract lags from buffer
        l1 = history[-1]
        l7 = history[-7]
        r7 = np.mean(history[-7:])
        l364 = history[-364]
        
        # Predict
        p = model.predict(pd.DataFrame([[l1, l7, r7, l364]], columns=features))[0]
        p = max(0, p)
        preds.append(p)
        history.append(p)
        
    mae_exp = np.abs(np.array(preds) - test_df['Revenue'].values).mean()
    
    print(f"\n--- Recursive Test Results (2022) ---")
    print(f"MAE Base (Current Pipeline) : {mae_base:,.0f}")
    print(f"MAE Exp (Recursive Lag 364) : {mae_exp:,.0f}")
    print(f"Improvement                 : {mae_base - mae_exp:,.0f}")

if __name__ == "__main__":
    verify_yoy_recursive()
