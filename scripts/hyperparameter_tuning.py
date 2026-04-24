import optuna
import pandas as pd
import numpy as np
import logging
import sys
import json
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

# Setup Logging
log_file = PROJECT_ROOT / "logs" / "tuning.log"
os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
Config.initialize_dirs()

# Load data once outside objective to save IO
logger.info("Loading sales data...")
raw_sales_global = pd.read_parquet(Config.SALES_TRAIN_FILE)
raw_sales_global['Date'] = pd.to_datetime(raw_sales_global['Date'])

def objective(trial):
    # 1. Define Search Space
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'n_estimators': 2000 # Fixed for consistency
    }
    
    logger.info(f"Trial {trial.number} started...")
    
    # 2. Run 3-Fold Walk-Forward CV
    folds = [
        {'train_max_year': 2019, 'test_start': 2020, 'test_end': 2020, 'weight': 0.2},
        {'train_max_year': 2020, 'test_start': 2021, 'test_end': 2021, 'weight': 0.3},
        {'train_max_year': 2021, 'test_start': 2022, 'test_end': 2022, 'weight': 0.5},
    ]
    
    total_weighted_mae = 0
    
    for fold in folds:
        train_end_date = pd.to_datetime(f"{fold['train_max_year']}-12-31")
        test_start_date = pd.to_datetime(f"{fold['test_start']}-01-01")
        test_end_date = pd.to_datetime(f"{fold['test_end']}-12-31")
        
        train_df = raw_sales_global[raw_sales_global['Date'] <= train_end_date].copy()
        test_df = raw_sales_global[(raw_sales_global['Date'] >= test_start_date) & (raw_sales_global['Date'] <= test_end_date)].copy()
        
        pipeline = ForecastingPipeline()
        # Inject params into both models
        pipeline.revenue_pipeline.named_steps['model'].set_params(**params)
        pipeline.cogs_pipeline.named_steps['model'].set_params(**params)
        
        pipeline.fit(train_df)
        predictions = pipeline.predict(test_df[['Date']])
        
        mae_rev = mean_absolute_error(test_df['Revenue'], predictions['Revenue'])
        mae_cogs = mean_absolute_error(test_df['COGS'], predictions['COGS'])
        
        total_weighted_mae += (mae_rev + mae_cogs) * fold['weight']
        
    logger.info(f"Trial {trial.number} finished. Weighted Total MAE: {total_weighted_mae:,.0f}")
    return total_weighted_mae

def run_tuning(n_trials=100):
    logger.info(f"Starting Bayesian Optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction='minimize', study_name="lgbm_tuning")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Tuning Complete!")
    logger.info(f"Best Trial Score: {study.best_value:,.0f}")
    logger.info(f"Best Params: {json.dumps(study.best_params, indent=2)}")
    
    # Save best params to a file using Config
    best_params_path = Config.MODEL_DIR / "best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    logger.info(f"Best parameters saved to {best_params_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of tuning trials')
    args = parser.parse_args()
    
    run_tuning(n_trials=args.n_trials)
