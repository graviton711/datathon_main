import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def verify_yoy_lag():
    print("Starting YoY Lag (rev_lag_364) Verification...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    folds = [
        (2013, 2019, 2020),
        (2013, 2020, 2021),
        (2013, 2021, 2022)
    ]
    
    results = []
    
    for (start_year, train_end, test_year) in folds:
        print(f"\nEvaluating Fold: Test {test_year}...")
        train_df = sales[(sales['Date'].dt.year >= start_year) & (sales['Date'].dt.year <= train_end)].copy()
        test_df = sales[sales['Date'].dt.year == test_year].copy()
        
        # --- Baseline ---
        pipe_base = ForecastingPipeline()
        pipe_base.fit(train_df)
        forecast_base = pipe_base.predict(test_df[['Date']])
        mae_base = np.abs(forecast_base['Revenue'].values - test_df['Revenue'].values).mean()
        
        # --- Experimental (Add rev_lag_364) ---
        pipe_exp = ForecastingPipeline()
        
        # Override _add_lags to include 364
        def custom_add_lags(df):
            df = df.copy()
            df['rev_lag_1'] = df['Revenue'].shift(1)
            df['rev_lag_7'] = df['Revenue'].shift(7)
            df['rev_roll_7'] = df['Revenue'].shift(1).rolling(7).mean()
            df['rev_lag_364'] = df['Revenue'].shift(364)
            return df
            
        pipe_exp._add_lags = custom_add_lags
        pipe_exp.lag_features = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7', 'rev_lag_364']
        pipe_exp.model_feature_order = pipe_exp.lag_features + pipe_exp.features
        
        # We also need to override predict because it manages its own buffer
        original_predict = pipe_exp.predict
        def custom_predict(df_horizon):
            # For simplicity in verification, we'll just use a slightly modified version of the original
            # but we need to ensure the buffer handles 364 lags
            return original_predict(df_horizon) # This will fail if we don't update predict's internal logic
            
        # Actually, let's just update the core pipeline.py temporarily for the test
        # or monkeypatch the internal recursive loop.
        # Given the complexity, I will just run a 1-shot (non-recursive) test first 
        # to see if the signal has ANY power in 1-step ahead prediction.
        
        print(f"  MAE Base: {mae_base:,.0f}")
        print(f"  MAE Exp : {mae_exp:,.0f} (Change: {mae_exp - mae_base:,.0f})")
        results.append({'fold': test_year, 'base': mae_base, 'exp': mae_exp})

    summary = pd.DataFrame(results)
    avg_base = summary['base'].mean()
    avg_exp = summary['exp'].mean()
    print(f"\n--- Final Results ---")
    print(f"Average Base MAE: {avg_base:,.0f}")
    print(f"Average Exp MAE : {avg_exp:,.0f}")
    print(f"Net Improvement : {avg_base - avg_exp:,.0f}")

if __name__ == "__main__":
    verify_yoy_lag()
