import pandas as pd
import numpy as np
import lightgbm as lgb
from src.data.loader import loader
from src.features.temporal import add_time_features
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("diagnostic")

def run_diagnostic():
    # Load data
    df = loader.get_merged_data()
    df = add_time_features(df, date_col='Date')
    
    # Split into Fold 1 (Train < 2022)
    train_end = pd.to_datetime('2021-12-31')
    test_start = pd.to_datetime('2022-01-01')
    test_end = pd.to_datetime('2022-06-30')
    
    features = ['year', 'month', 'day', 'dayofweek', 'is_weekend']
    target = 'Revenue'
    
    train_df = df[df['Date'] <= train_end]
    test_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Train
    model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, importance_type='gain')
    model.fit(X_train, y_train)
    
    # Check importance
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance (Gain) ===")
    print(importances)
    
    # Check predictions vs reality scale
    preds = model.predict(X_test)
    print(f"\nActual Mean: {y_test.mean():,.0f}")
    print(f"Pred Mean:   {preds.mean():,.0f}")
    print(f"Actual Std:  {y_test.std():,.0f}")
    
    # Correlation Check
    corr = np.corrcoef(y_test, preds)[0, 1]
    print(f"Test Correlation: {corr:.3f}")

if __name__ == "__main__":
    run_diagnostic()
