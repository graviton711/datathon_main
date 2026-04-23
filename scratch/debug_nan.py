import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def debug_nan():
    print("Debugging NaN issues in recursive forecast...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Train on everything until 2022
    train_df = sales[sales['Date'].dt.year <= 2022].copy()
    test_df = pd.DataFrame({'Date': pd.date_range('2023-01-01', '2024-07-01', freq='D')})
    
    pipeline = ForecastingPipeline()
    pipeline.fit(train_df)
    
    # Run predict and monitor
    print("\nStarting prediction...")
    try:
        forecast = pipeline.predict(test_df)
        print(f"Forecast complete. NaNs found: {forecast['Revenue'].isna().sum()}")
        if forecast['Revenue'].isna().any():
            print("First NaN at:")
            print(forecast[forecast['Revenue'].isna()].head(1))
    except Exception as e:
        print(f"Prediction crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_nan()
