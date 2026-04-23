import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def investigate_underprediction():
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Train Pipeline to get internal states
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    print("\n--- Internal State Investigation ---")
    print(f"Base Scale (2022 Median): {pipeline.base_scale_rev:,.2f}")
    
    # Check Event Score for 12/31
    event_map = pipeline.revenue_pipeline.named_steps['features'].event_score_map
    score_31_12 = event_map.get((12, 31), 1.0)
    print(f"Event Score for 31/12: {score_31_12:.2f}x")
    
    # Check Momentum
    print(f"Momentum (Base): {pipeline.momentum['base']:.3f}x")
    print(f"Momentum (Event): {pipeline.momentum['event']:.3f}x")
    
    # Calculate Theoretical Prediction for 31/12/2023
    # Prediction = Raw_Norm * Base_Scale * Momentum
    # Assuming Raw_Norm is around 1.0 (average day) or higher for events
    theoretical_raw_norm = 1.0 * score_31_12 # Simple estimation
    pred_2023 = theoretical_raw_norm * pipeline.base_scale_rev * pipeline.momentum['event']
    print(f"Theoretical 31/12/2023 Prediction: {pred_2023:,.2f}")
    
    # Compare with historical Dec 31sts
    yearly_31_12 = sales[(sales['Date'].dt.month == 12) & (sales['Date'].dt.day == 31)]
    print("\n--- Historical 31/12 Revenue ---")
    print(yearly_31_12[['Date', 'Revenue']].sort_values('Date'))

if __name__ == "__main__":
    investigate_underprediction()
