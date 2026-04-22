import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BaselineFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A standard Scikit-Learn compatible transformer for baseline feature extraction.
    This creates the absolute minimum features needed for a time-series baseline.
    """
    def __init__(self, date_col='Date'):
        self.date_col = date_col
        self.start_date_ref = None

    def fit(self, X, y=None):
        # Memorize the start date from the training set to calculate 'days_from_start'
        if self.date_col in X.columns:
            self.start_date_ref = pd.to_datetime(X[self.date_col]).min()
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            
        # 1. Basic Time Features
        X['year'] = X[self.date_col].dt.year
        X['month'] = X[self.date_col].dt.month
        X['day'] = X[self.date_col].dt.day
        X['day_of_week'] = X[self.date_col].dt.dayofweek
        X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
        
        # 2. Linear Trend Placeholder
        if self.start_date_ref is not None:
            X['days_from_start'] = (X[self.date_col] - self.start_date_ref).dt.days
        else:
            X['days_from_start'] = 0
            
        # Drop the original date column for training (LightGBM requires numeric/categorical)
        X = X.drop(columns=[self.date_col])
        return X

    def get_feature_names(self):
        return ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'days_from_start']
