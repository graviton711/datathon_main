import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def run_inventory_experiment():
    print("--- Testing Inventory Signal (units_received) ---")
    
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Load and Prepare Inventory
    inventory = pd.read_csv(Config.RAW_DATA_DIR / "inventory.csv")
    inventory['snapshot_date'] = pd.to_datetime(inventory['snapshot_date'])
    # Aggregate monthly units to dates
    monthly_units = inventory.groupby(['snapshot_date'])['units_received'].sum().reset_index()
    monthly_units['year'] = monthly_units['snapshot_date'].dt.year
    monthly_units['month'] = monthly_units['snapshot_date'].dt.month
    
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    sales = pd.merge(sales, monthly_units[['year', 'month', 'units_received']], on=['year', 'month'], how='left')
    sales['units_received'] = sales['units_received'].fillna(0)
    
    # Normalize units_received by its annual median to keep it stationary
    annual_unit_medians = sales.groupby('year')['units_received'].median().to_dict()
    sales['units_norm'] = sales['units_received'] / sales['year'].map(annual_unit_medians).replace(0, 1.0)

    # 2. Add units_norm to features (Mocking by modifying the extractor dynamically)
    print("Injecting units_norm into Pipeline...")
    
    train_df = sales[sales['Date'] <= '2021-12-31'].copy()
    test_df = sales[(sales['Date'] >= '2022-01-01') & (sales['Date'] <= '2022-12-31')].copy()
    
    # We need to ensure the pipeline sees this new column
    # Since the pipeline uses BaselineFeatureExtractor, we'll need to wrap it or modify it
    
    pipeline = ForecastingPipeline()
    # Mocking the transform method to include units_norm
    original_transform = pipeline.revenue_pipeline.named_steps['features'].transform
    
    def custom_transform(X):
        # Merge units_norm back into X based on Date
        X_dates = X[['Date']].copy()
        X_dates['year'] = X_dates['Date'].dt.year
        X_dates['month'] = X_dates['Date'].dt.month
        X_new = pd.merge(X_dates, sales[['year', 'month', 'units_norm']].drop_duplicates(), on=['year', 'month'], how='left')
        
        res = original_transform(X)
        res['units_norm'] = X_new['units_norm'].values
        return res

    pipeline.revenue_pipeline.named_steps['features'].transform = custom_transform
    pipeline.cogs_pipeline.named_steps['features'].transform = custom_transform
    
    # Update model feature order
    pipeline.model_feature_order.append('units_norm')
    
    # Fit & Predict
    pipeline.fit(train_df)
    preds = pipeline.predict(test_df[['Date']])
    
    mae_rev = mean_absolute_error(test_df['Revenue'], preds['Revenue'])
    print(f"\nMAE with units_norm: {mae_rev:,.0f} (Baseline was ~608k)")
    
    # Feature Importance
    importances = pipeline.revenue_pipeline.named_steps['model'].feature_importances_
    feat_names = pipeline.model_feature_order
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
    print("\nTop 5 Features:")
    print(imp_df.head(5))

if __name__ == "__main__":
    run_inventory_experiment()
