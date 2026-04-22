import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.config import Config
from src.features.builder import BaselineFeatureExtractor

class ForecastingPipeline:
    """
    Standard OOP Pipeline orchestrator for training and inference.
    """
    def __init__(self):
        self.features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'days_from_start']
        self.categorical_features = ['month', 'day_of_week', 'is_weekend']
        
        # Scikit-Learn Pipeline for Revenue
        self.revenue_pipeline = Pipeline([
            ('features', BaselineFeatureExtractor(date_col='Date')),
            ('model', lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                objective='regression',
                random_state=42,
                verbose=-1
            ))
        ])
        
        # Scikit-Learn Pipeline for COGS Ratio
        self.cogs_pipeline = Pipeline([
            ('features', BaselineFeatureExtractor(date_col='Date')),
            ('model', lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                objective='regression',
                random_state=42,
                verbose=-1
            ))
        ])

    def fit(self, df: pd.DataFrame):
        print("Training Baseline Revenue Model...")
        # Lấy X và y
        X = df[['Date']].copy()
        y_rev = df['Revenue']
        y_cogs_ratio = df['COGS'] / (df['Revenue'] + 1e-6)
        
        # Calculate time-decay weights inside fit
        start_date_ref = df['Date'].min()
        days_from_start = (df['Date'] - start_date_ref).dt.days
        max_days = days_from_start.max()
        sample_weights = np.exp((days_from_start - max_days) / 365.0)
        
        # Train pipelines
        # LightGBM categorical features must be passed in fit_params
        fit_params = {
            'model__categorical_feature': self.categorical_features,
            'model__sample_weight': sample_weights
        }
        
        self.revenue_pipeline.fit(X, y_rev, **fit_params)
        
        print("Training Baseline COGS Ratio Model...")
        self.cogs_pipeline.fit(X, y_cogs_ratio, **fit_params)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Generating forecast...")
        X = df[['Date']].copy()
        
        preds_rev = self.revenue_pipeline.predict(X)
        preds_ratio = self.cogs_pipeline.predict(X)
        
        result = df[['Date']].copy()
        result['Revenue'] = np.maximum(0, preds_rev)
        result['COGS'] = result['Revenue'] * np.clip(preds_ratio, 0.0, 1.0)
        
        return result

def run_baseline():
    print("--- Starting Baseline Pipeline ---")
    
    # 1. Load Data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Initialize and Train Pipeline
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    # 3. Predict Horizon
    horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': horizon_dates})
    
    submission = pipeline.predict(test_df)
    
    # 4. Save
    out_path = Config.SUBMISSION_DIR / 'submission.csv'
    submission.to_csv(out_path, index=False)
    print(f"--- Done! Saved Baseline to {out_path} ---")

if __name__ == "__main__":
    run_baseline()
