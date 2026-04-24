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

def run_urban_blowout_verification():
    print("\n--- Verifying Urban Blowout Shape & Relaxed Clipping ---")
    raw_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    raw_sales['Date'] = pd.to_datetime(raw_sales['Date'])
    
    # Test on 2021 (Odd Year)
    test_year = 2021
    train = raw_sales[raw_sales['Date'].dt.year < test_year].copy()
    test = raw_sales[raw_sales['Date'].dt.year == test_year].copy()
    
    p = ForecastingPipeline()
    p._validate_feature_contract = lambda x: None
    p.fit(train)
    
    # 1. Prepare Features with Urban Blowout Shape
    def add_urban_features(df, X):
        df = df.copy()
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        
        # Days since July 30th of the current year (if odd year)
        def get_days_since(row):
            if row['year'] % 2 == 0: return 0
            start_date = pd.Timestamp(year=row['year'], month=7, day=30)
            diff = (row['Date'] - start_date).days
            if 0 <= diff <= 35: # Up to end of August
                return diff + 1
            return 0
            
        X['urban_days'] = df.apply(get_days_since, axis=1).values
        return X

    X_train = p.revenue_pipeline.named_steps['features'].transform(train)
    X_test = p.revenue_pipeline.named_steps['features'].transform(test)
    
    X_train = add_urban_features(train, X_train)
    X_test = add_urban_features(test, X_test)
    
    # 2. Relaxed Clipping (q999 instead of q99)
    y_train_cogs = train['COGS'] / (train['Revenue'] + 1e-6)
    q999 = y_train_cogs.quantile(0.999)
    print(f"Relaxed Clip Upper Bound (q999): {q999:.4f}")
    
    # 3. Train & Predict
    cogs_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
    cogs_model.fit(X_train, y_train_cogs)
    
    preds_cogs_ratio = np.clip(cogs_model.predict(X_test), 0.0, q999)
    final_cogs = preds_cogs_ratio * test['Revenue'].values
    
    # 4. Results for August 2021
    aug_mask = (test['Date'].dt.month == 8).values
    aug_mae = mean_absolute_error(test.loc[aug_mask, 'COGS'], final_cogs[aug_mask])
    
    print(f"\nResults for August 2021:")
    print(f"  August COGS MAE (with Shape + Relaxed Clip): {aug_mae:,.0f}")
    print(f"  Baseline was: 1,334,310")
    print(f"  V2 (Binary Flag) was: 220,080")

if __name__ == "__main__":
    run_urban_blowout_verification()
