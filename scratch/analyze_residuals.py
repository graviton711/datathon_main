import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline

def analyze_residuals():
    print("Starting Residual Analysis (Error Audit)...")
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Train on 2013-2021, Test on 2022
    train_df = sales[sales['Date'].dt.year < 2022].copy()
    test_df = sales[sales['Date'].dt.year == 2022].copy()
    
    pipeline = ForecastingPipeline()
    pipeline.fit(train_df)
    forecast = pipeline.predict(test_df[['Date']])
    
    # Calculate daily error
    audit = test_df.copy()
    audit['Pred'] = forecast['Revenue'].values
    audit['Error'] = audit['Pred'] - audit['Revenue']
    audit['Abs_Error'] = audit['Error'].abs()
    
    # Top 20 Failures
    worst_days = audit.sort_values('Abs_Error', ascending=False).head(20)
    
    print("\n--- Top 20 Prediction Failures (2022) ---")
    print(worst_days[['Date', 'Revenue', 'Pred', 'Error', 'Abs_Error']])
    
    # Check if there's a pattern in dates
    worst_days['month'] = worst_days['Date'].dt.month
    worst_days['day'] = worst_days['Date'].dt.day
    
    print("\nCommon Months for Failure:")
    print(worst_days['month'].value_counts())
    
    print("\nCommon Days for Failure:")
    print(worst_days['day'].value_counts())

if __name__ == "__main__":
    analyze_residuals()
