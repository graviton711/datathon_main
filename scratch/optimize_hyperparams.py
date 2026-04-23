import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.pipeline import ForecastingPipeline
from src.config import Config

def objective(trial):
    # 1. Suggest Hyperparameters
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'objective': 'regression',
        'random_state': 42,
        'verbose': -1
    }
    
    # 2. Setup Evaluation (Walk-Forward 3-Fold)
    DATA_DIR = Path("data/processed")
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    folds = [
        (2013, 2019, 2020),
        (2013, 2020, 2021),
        (2013, 2021, 2022)
    ]
    weights = [0.2, 0.3, 0.5]
    
    fold_maes = []
    
    for (start_year, train_end, test_year), weight in zip(folds, weights):
        train_df = sales[(sales['Date'].dt.year >= start_year) & (sales['Date'].dt.year <= train_end)].copy()
        test_df = sales[sales['Date'].dt.year == test_year].copy()
        
        # Initialize Pipeline with trial params
        pipeline = ForecastingPipeline()
        # Manually override the model in the pipeline
        pipeline.revenue_pipeline.named_steps['model'].set_params(**params)
        pipeline.cogs_pipeline.named_steps['model'].set_params(**params)
        
        # Fit and Evaluate
        pipeline.fit(train_df)
        
        # Predict 
        forecast = pipeline.predict(test_df[['Date']])
        
        # Calculate Revenue MAE
        mae = np.abs(forecast['Revenue'] - test_df['Revenue']).mean()
        fold_maes.append(mae * weight)
        
    return sum(fold_maes)

def run_optimization():
    print("Starting Bayesian Optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # Low trials for quick verification
    
    print("\n--- Best Trial ---")
    print(f"  Value (Weighted Rev MAE): {study.best_value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    # Save best params to a scratch file
    with open('scratch/best_params.txt', 'w') as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    run_optimization()
