import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def verify_procurement_signal():
    print("Starting Procurement Signal Verification...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    inv = pd.read_parquet(DATA_DIR / "inventory.parquet")
    
    # 1. Create Procurement Intensity Profile (Monthly)
    print("Calculating Procurement Intensity Profile...")
    inv['month'] = pd.to_datetime(inv['snapshot_date']).dt.month
    proc_profile = inv.groupby('month')['units_received'].mean().to_dict()
    
    # 2. Setup Cross-Validation (Fold 3 - 2022 only for speed, or all 3)
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
        
        # --- Baseline (Current Pipeline) ---
        pipe_base = ForecastingPipeline()
        pipe_base.fit(train_df)
        forecast_base = pipe_base.predict(test_df[['Date']])
        mae_base = np.abs(forecast_base['Revenue'].values - test_df['Revenue'].values).mean()
        
        # --- Experimental (With Procurement Intensity) ---
        # We manually inject the procurement feature into the transform phase
        # To do this cleanly, we'll monkey-patch the get_feature_names and transform
        original_get_names = pipe_base.revenue_pipeline.named_steps['features'].get_feature_names
        original_transform = pipe_base.revenue_pipeline.named_steps['features'].transform
        
        def new_get_names():
            return original_get_names() + ['procurement_intensity']
        
        def new_transform(X):
            X_out = original_transform(X)
            X_out['procurement_intensity'] = X_out['month'].map(proc_profile).fillna(0.0)
            return X_out
        
        # Re-initialize for experiment
        pipe_exp = ForecastingPipeline()
        # Apply the monkey-patch to both revenue and cogs pipelines
        for p in [pipe_exp.revenue_pipeline, pipe_exp.cogs_pipeline]:
            p.named_steps['features'].get_feature_names = new_get_names
            p.named_steps['features'].transform = new_transform
        
        # Re-set feature order
        pipe_exp.features = new_get_names()
        pipe_exp.model_feature_order = pipe_exp.lag_features + pipe_exp.features
        
        pipe_exp.fit(train_df)
        forecast_exp = pipe_exp.predict(test_df[['Date']])
        mae_exp = np.abs(forecast_exp['Revenue'].values - test_df['Revenue'].values).mean()
        
        print(f"  MAE Base: {mae_base:,.0f}")
        print(f"  MAE Exp : {mae_exp:,.0f} (Change: {mae_exp - mae_base:,.0f})")
        
        results.append({'fold': test_year, 'base': mae_base, 'exp': mae_exp})
    
    # Summary
    summary = pd.DataFrame(results)
    avg_base = summary['base'].mean()
    avg_exp = summary['exp'].mean()
    print(f"\n--- Final Results ---")
    print(f"Average Base MAE: {avg_base:,.0f}")
    print(f"Average Exp MAE : {avg_exp:,.0f}")
    print(f"Net Improvement : {avg_base - avg_exp:,.0f}")

if __name__ == "__main__":
    verify_procurement_signal()
