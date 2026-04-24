import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def run_cogs_verification(use_odd_feature=False):
    print(f"\n--- Verifying COGS with is_odd_year={use_odd_feature} ---")
    raw_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    raw_sales['Date'] = pd.to_datetime(raw_sales['Date'])
    
    folds = [
        {'train_max': 2020, 'test': 2021, 'weight': 1.0}, # Focus on Odd Year 2021
    ]
    
    for fold in folds:
        train = raw_sales[raw_sales['Date'].dt.year <= fold['train_max']].copy()
        test = raw_sales[raw_sales['Date'].dt.year == fold['test']].copy()
        
        p = ForecastingPipeline()
        p._validate_feature_contract = lambda x: None
        p.fit(train)
        
        # Get baseline features
        X_train = p.revenue_pipeline.named_steps['features'].transform(train)
        X_test = p.revenue_pipeline.named_steps['features'].transform(test)
        
        if use_odd_feature:
            # Add the Odd Year Signal
            X_train['is_odd_year'] = (train['Date'].dt.year % 2 != 0).astype(int)
            X_test['is_odd_year'] = (test['Date'].dt.year % 2 != 0).astype(int)
            
            # Relax Clipping for the verification
            p.cogs_ratio_clip = (0.0, 2.0)
        
        # Train COGS Model
        y_train_cogs = train['COGS'] / (train['Revenue'] + 1e-6)
        cogs_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
        cogs_model.fit(X_train, y_train_cogs)
        
        # Predict
        preds_cogs_ratio = cogs_model.predict(X_test)
        if use_odd_feature:
            preds_cogs_ratio = np.clip(preds_cogs_ratio, 0.0, 2.0)
        else:
            preds_cogs_ratio = np.clip(preds_cogs_ratio, p.cogs_ratio_clip[0], p.cogs_ratio_clip[1])
            
        final_cogs = preds_cogs_ratio * test['Revenue'].values
        
        # Calculate MAE for August specifically
        test['pred_cogs'] = final_cogs
        aug_test = test[test['Date'].dt.month == 8]
        aug_mae = mean_absolute_error(aug_test['COGS'], aug_test['pred_cogs'])
        total_mae = mean_absolute_error(test['COGS'], final_cogs)
        
        print(f"Fold {fold['test']} (Odd Year):")
        print(f"  August COGS MAE: {aug_mae:,.0f}")
        print(f"  Overall COGS MAE: {total_mae:,.0f}")
        
        if use_odd_feature:
            # Check the predicted ratio for August 2021
            avg_pred_ratio = np.mean(preds_cogs_ratio[test['Date'].dt.month == 8])
            print(f"  Average Predicted Ratio for August: {avg_pred_ratio:.4f}")

if __name__ == "__main__":
    run_cogs_verification(use_odd_feature=False)
    run_cogs_verification(use_odd_feature=True)
