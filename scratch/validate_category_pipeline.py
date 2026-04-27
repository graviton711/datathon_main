import pandas as pd
import numpy as np
from src.config import Config
from src.training.pipeline import ForecastingPipeline

def mape(actual, pred):
    mask = actual > 0
    return (np.abs(actual[mask] - pred[mask]) / actual[mask]).mean() * 100

def validate():
    print("--- Starting Category-Specific Pipeline Validation ---")
    
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Split: Train on data before 2022, Validate on 2022 (to see if category momentum helps)
    train_df = sales[sales['Date'] < '2022-01-01'].copy()
    val_df = sales[sales['Date'] >= '2022-01-01'].copy()
    
    print(f"Training on {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"Validating on {val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
    
    # 2. Train Pipeline
    pipeline = ForecastingPipeline()
    pipeline.fit(train_df)
    
    # 3. Predict Validation Period
    horizon_df = val_df[['Date']].copy()
    preds = pipeline.predict(horizon_df)
    
    # 4. Compare
    results = pd.merge(val_df[['Date', 'Revenue', 'COGS']], preds, on='Date', suffixes=('_act', '_pred'))
    
    rev_mape = mape(results['Revenue_act'], results['Revenue_pred'])
    cogs_mape = mape(results['COGS_act'], results['COGS_pred'])
    
    print(f"\n--- Validation Results ---")
    print(f"Revenue MAPE: {rev_mape:.2f}%")
    print(f"COGS MAPE:    {cogs_mape:.2f}%")
    
    # Compare with a simple constant momentum baseline if possible
    # (Just as a sanity check)
    
    return results

if __name__ == "__main__":
    validate()
