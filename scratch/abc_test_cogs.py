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

def run_variant_test(variant_name):
    raw_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    raw_sales['Date'] = pd.to_datetime(raw_sales['Date'])
    
    # We'll test on 2021 (Odd) and 2022 (Even)
    results = {}
    
    for test_year in [2021, 2022]:
        train = raw_sales[raw_sales['Date'].dt.year < test_year].copy()
        test = raw_sales[raw_sales['Date'].dt.year == test_year].copy()
        
        p = ForecastingPipeline()
        p._validate_feature_contract = lambda x: None
        p.fit(train)
        
        X_train = p.revenue_pipeline.named_steps['features'].transform(train)
        X_test = p.revenue_pipeline.named_steps['features'].transform(test)
        
        # Apply Variant Logic
        for X, source_df in [(X_train, train), (X_test, test)]:
            yr = source_df['Date'].dt.year
            mo = source_df['Date'].dt.month
            if variant_name == 'V1':
                X['odd_signal'] = (yr % 2 != 0).astype(int)
            elif variant_name == 'V2':
                X['odd_signal'] = ((yr % 2 != 0) & (mo == 8)).astype(int)
            else:
                X['odd_signal'] = 0 # Baseline
        
        # Train & Predict COGS Ratio
        y_train_cogs = train['COGS'] / (train['Revenue'] + 1e-6)
        cogs_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
        cogs_model.fit(X_train, y_train_cogs)
        
        preds_cogs_ratio = np.clip(cogs_model.predict(X_test), 0.0, 2.0)
        test['pred_cogs'] = preds_cogs_ratio * test['Revenue'].values
        
        # Analysis
        aug_mask = test['Date'].dt.month == 8
        aug_mae = mean_absolute_error(test[aug_mask]['COGS'], test[aug_mask]['pred_cogs'])
        non_aug_mae = mean_absolute_error(test[~aug_mask]['COGS'], test[~aug_mask]['pred_cogs'])
        
        results[test_year] = {'aug_mae': aug_mae, 'non_aug_mae': non_aug_mae}
        
    return results

if __name__ == "__main__":
    print("--- HEAD-TO-HEAD VARIANT TEST ---")
    final_table = []
    for v in ['Baseline', 'V1', 'V2']:
        res = run_variant_test(v)
        final_table.append({
            'Variant': v,
            '2021_Aug_COGS_MAE': res[2021]['aug_mae'],
            '2021_Other_COGS_MAE': res[2021]['non_aug_mae'],
            '2022_Aug_COGS_MAE': res[2022]['aug_mae'],
            '2022_Other_COGS_MAE': res[2022]['non_aug_mae'],
        })
    
    df_res = pd.DataFrame(final_table)
    print("\nComparison Table:")
    print(df_res.to_string(index=False))
