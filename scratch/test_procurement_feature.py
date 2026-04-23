import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.features.builder import BaselineFeatureExtractor
from src.config import Config

def test_feature():
    print("Loading sales data...")
    try:
        sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    except:
        sales = pd.read_csv(Config.DATA_DIR / "raw" / "sales.csv")
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Simple split: Train <= 2021, Test = 2022 (Fold 3 equivalent)
    train_sales = sales[sales['Date'].dt.year <= 2021].copy()
    test_sales = sales[sales['Date'].dt.year == 2022].copy()
    
    # 1. Baseline Features Extraction
    print("Extracting base features...")
    extractor = BaselineFeatureExtractor(date_col='Date')
    extractor.fit(train_sales, y=train_sales['Revenue'])
    
    X_train_base = extractor.transform(train_sales)
    X_test_base = extractor.transform(test_sales)
    
    y_train = train_sales['Revenue']
    y_test = test_sales['Revenue']
    
    # 2. Add Procurement Feature
    print("Calculating procurement profile...")
    try:
        inv = pd.read_parquet(Config.DATA_DIR / "processed" / "inventory.parquet")
    except:
        inv = pd.read_csv(Config.DATA_DIR / "raw" / "inventory.csv")
        
    inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'])
    
    # Prevent leakage: only use inventory up to 2021 to calculate profile
    inv_train = inv[inv['snapshot_date'].dt.year <= 2021].copy()
    inv_train['year'] = inv_train['snapshot_date'].dt.year
    inv_train['month'] = inv_train['snapshot_date'].dt.month
    
    monthly_inbound = inv_train.groupby(['year', 'month'])['units_received'].sum().reset_index()
    monthly_inbound.columns = ['year', 'month', 'units_received']
    yearly_inbound = monthly_inbound.groupby('year')['units_received'].sum().reset_index()
    yearly_inbound.columns = ['year', 'year_total']
    
    merged = pd.merge(monthly_inbound, yearly_inbound, on='year')
    merged['inbound_share'] = merged['units_received'] / (merged['year_total'] + 1e-6)
    
    # Create the mapping dictionary
    profile = merged.groupby('month')['inbound_share'].median().to_dict()
    
    X_train_new = X_train_base.copy()
    X_test_new = X_test_base.copy()
    
    X_train_new['procurement_share'] = X_train_new['month'].map(profile)
    X_test_new['procurement_share'] = X_test_new['month'].map(profile)
    
    # Train Base Model
    # Setting simple parameters for fast external validation
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
    
    cat_features = ['month', 'day_of_week', 'is_wednesday', 'is_weekend', 'is_payday_start', 'is_payday_end', 'is_quarter_end']
    cats_to_use = [c for c in cat_features if c in X_train_base.columns]
    
    print("Training Base Model...")
    model_base = lgb.LGBMRegressor(**params)
    model_base.fit(X_train_base, y_train, categorical_feature=cats_to_use)
    preds_base = model_base.predict(X_test_base)
    mae_base = mean_absolute_error(y_test, preds_base)
    
    print("Training Enhanced Model (with Procurement Share)...")
    model_new = lgb.LGBMRegressor(**params)
    model_new.fit(X_train_new, y_train, categorical_feature=cats_to_use)
    preds_new = model_new.predict(X_test_new)
    mae_new = mean_absolute_error(y_test, preds_new)
    
    print("\n=== EXTERNAL VALIDATION RESULTS (Fold 3: 2022) ===")
    print(f"Base MAE           : {mae_base:,.0f}")
    print(f"Enhanced MAE       : {mae_new:,.0f}")
    diff = mae_base - mae_new
    if diff > 0:
        print(f"Improvement        : +{diff:,.0f} (LOWER is better)")
    else:
        print(f"Degradation        : {diff:,.0f} (Model got worse)")
    
    # Feature Importance
    importance = pd.DataFrame({'feature': X_train_new.columns, 'importance': model_new.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances (Enhanced Model):")
    print(importance.head(10).to_string(index=False))
    
    print(f"\nProcurement Share Rank: {importance[importance['feature'] == 'procurement_share'].index[0] + 1} out of {len(importance)}")

if __name__ == "__main__":
    test_feature()
