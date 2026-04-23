import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def verify_yoy_normalization():
    print("Verifying YoY Seasonality Normalization (Loop-Free)...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Use Fold 3: Train until 2021, Test 2022
    train_df = sales[sales['Date'].dt.year < 2022].copy()
    test_df = sales[sales['Date'].dt.year == 2022].copy()
    
    # --- Baseline ---
    pipe_base = ForecastingPipeline()
    pipe_base.fit(train_df)
    forecast_base = pipe_base.predict(test_df[['Date']])
    mae_base = np.abs(forecast_base['Revenue'].values - test_df['Revenue'].values).mean()
    
    # --- Experimental: YoY Normalization ---
    print("Training Experimental Model (YoY Norm)...")
    
    # 1. Pre-calculate Reference Scale (Rolling 30d of Last Year)
    full_data = sales.sort_values('Date').copy()
    full_data['ref_scale'] = full_data['Revenue'].shift(364).rolling(30, min_periods=1).mean()
    # Fill early years with annual median
    full_data['year'] = full_data['Date'].dt.year
    annual_medians = full_data.groupby('year')['Revenue'].median()
    full_data['ref_scale'] = full_data['ref_scale'].fillna(full_data['year'].map(annual_medians))
    
    full_data['target_norm'] = full_data['Revenue'] / (full_data['ref_scale'] + 1e-6)
    
    # Prepare Train
    train_ext = full_data[full_data['Date'].dt.year < 2022].copy()
    train_ext['rev_lag_1'] = train_ext['target_norm'].shift(1)
    train_ext['rev_lag_7'] = train_ext['target_norm'].shift(7)
    train_ext['rev_roll_7'] = train_ext['target_norm'].shift(1).rolling(7).mean()
    train_ext = train_ext.dropna().reset_index(drop=True)
    
    features = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7']
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(train_ext[features], train_ext['target_norm'])
    
    # 2. Recursive Loop with STATIC ref_scale
    print("Running Recursive Loop (YoY Norm)...")
    # Buffer for target_norm
    history_norm = list(train_ext['target_norm'].values)
    preds_final = []
    
    test_dates = test_df['Date'].values
    for i in range(len(test_df)):
        l1 = history_norm[-1]
        l7 = history_norm[-7]
        r7 = np.mean(history_norm[-7:])
        
        # Predict the NORM value
        p_norm = model.predict(pd.DataFrame([[l1, l7, r7]], columns=features))[0]
        
        # Get the STATIC scale from last year
        # Note: test_df already contains the ref_scale in our full_data
        curr_date = test_dates[i]
        scale = full_data[full_data['Date'] == curr_date]['ref_scale'].values[0]
        
        p_final = max(0, p_norm * scale)
        preds_final.append(p_final)
        history_norm.append(p_norm)
        
    mae_exp = np.abs(np.array(preds_final) - test_df['Revenue'].values).mean()
    
    print(f"\n--- YoY Normalization Test Results (2022) ---")
    print(f"MAE Base (Annual Median Norm) : {mae_base:,.0f}")
    print(f"MAE Exp (YoY Seasonality Norm): {mae_exp:,.0f}")
    print(f"Improvement                   : {mae_base - mae_exp:,.0f}")

if __name__ == "__main__":
    verify_yoy_normalization()
