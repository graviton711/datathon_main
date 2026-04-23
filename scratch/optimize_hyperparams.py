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
    # 1. Suggest Hyperparameters (More conservative for recursive stability)
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 48),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.04),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 60),
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
        try:
            train_df = sales[(sales['Date'].dt.year >= start_year) & (sales['Date'].dt.year <= train_end)].copy()
            test_df = sales[sales['Date'].dt.year == test_year].copy().dropna(subset=['Revenue'])
            
            # Initialize Pipeline with trial params
            pipeline = ForecastingPipeline()
            pipeline.revenue_pipeline.named_steps['model'].set_params(**params)
            pipeline.cogs_pipeline.named_steps['model'].set_params(**params)
            
            pipeline.fit(train_df)
            forecast = pipeline.predict(test_df[['Date']])
            
            if forecast['Revenue'].isnull().any():
                print(f"  [Fold {test_year}] Found NaNs in forecast. Trial failed.")
                return 1e9
                
            mae = np.abs(forecast['Revenue'].values - test_df['Revenue'].values).mean()
            fold_maes.append(mae * weight)
        except Exception as e:
            print(f"  [Fold {test_year}] Crash: {e}")
            return 1e9
        
    final_score = sum(fold_maes)
    return final_score if not np.isnan(final_score) else 1e9

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
