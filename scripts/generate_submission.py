import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def generate_submission():
    print("--- Generating Submission File ---")
    
    # 1. Load All Training Data
    print("Loading training data...")
    train_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    train_sales['Date'] = pd.to_datetime(train_sales['Date'])
    
    # 2. Load Best Params
    best_params_path = PROJECT_ROOT / "models" / "best_params.json"
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    print(f"Using best parameters: {json.dumps(best_params, indent=2)}")
    
    # 3. Initialize and Fit Pipeline
    print("Fitting pipeline on full training data...")
    pipeline = ForecastingPipeline()
    # Inject optimized params
    pipeline.revenue_pipeline.named_steps['model'].set_params(**best_params)
    pipeline.cogs_pipeline.named_steps['model'].set_params(**best_params)
    
    pipeline.fit(train_sales)
    
    # 4. Generate Test Dates
    print("Generating predictions for 2023-01-01 to 2024-07-01...")
    test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': test_dates})
    
    # 5. Predict
    predictions = pipeline.predict(test_df)
    
    # 6. Format and Save Submission
    submission = predictions[['Date', 'Revenue', 'COGS']].copy()
    
    # Ensure formatting
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
    
    submission_path = PROJECT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission file generated successfully at: {submission_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Revenue Range: {submission['Revenue'].min():,.0f} to {submission['Revenue'].max():,.0f}")

if __name__ == "__main__":
    generate_submission()
