import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def verify_payday_elasticity():
    print("Starting Seasonal Payday Elasticity Verification...")
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
        
        # Calculate Lift Map from TRAIN data only to avoid leakage
        train_df['month'] = train_df['Date'].dt.month
        train_df['day'] = train_df['Date'].dt.day
        train_df['is_payday'] = (train_df['day'] >= 25) | (train_df['day'] <= 5)
        
        lift_profile = train_df.groupby(['month', 'is_payday'])['Revenue'].mean().unstack()
        payday_lift_map = (lift_profile[True] / lift_profile[False]).to_dict()
        
        # --- Baseline ---
        pipe_base = ForecastingPipeline()
        pipe_base.fit(train_df)
        forecast_base = pipe_base.predict(test_df[['Date']])
        mae_base = np.abs(forecast_base['Revenue'].values - test_df['Revenue'].values).mean()
        
        # --- Experimental ---
        pipe_exp = ForecastingPipeline()
        
        orig_get_names = pipe_exp.revenue_pipeline.named_steps['features'].get_feature_names
        orig_transform = pipe_exp.revenue_pipeline.named_steps['features'].transform
        
        def new_get_names():
            return orig_get_names() + ['payday_seasonal_intensity']
        
        def new_transform(X):
            X_out = orig_transform(X)
            # intensity = is_payday (0/1) * monthly_lift
            X_out['payday_seasonal_intensity'] = X_out['is_payday_start'] * X_out['month'].map(payday_lift_map).fillna(1.0)
            return X_out
            
        # Re-initialize for experiment
        pipe_exp = ForecastingPipeline()
        for p in [pipe_exp.revenue_pipeline, pipe_exp.cogs_pipeline]:
            p.named_steps['features'].get_feature_names = new_get_names
            p.named_steps['features'].transform = new_transform
        
        # Pre-fit to discover categories and set correct feature order
        pipe_exp.revenue_pipeline.named_steps['features'].fit(train_df, train_df['Revenue'])
        pipe_exp.features = pipe_exp.revenue_pipeline.named_steps['features'].get_feature_names()
        pipe_exp.model_feature_order = pipe_exp.lag_features + pipe_exp.features
        
        pipe_exp.fit(train_df)
        forecast_exp = pipe_exp.predict(test_df[['Date']])
        mae_exp = np.abs(forecast_exp['Revenue'].values - test_df['Revenue'].values).mean()
        
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
    verify_payday_elasticity()
