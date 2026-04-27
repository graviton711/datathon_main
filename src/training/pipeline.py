import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.features.builder import BaselineFeatureExtractor
from src.training.analyst import MarketAnalyst
from src.training.weighting import DataWeighting

class ForecastingPipeline:
    """
    Standard OOP Pipeline orchestrator for training and inference.
    """
    def __init__(self):
        self.lag_features = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7']
        self.features = [] # Will be populated dynamically by the extractor
        self.feature_cols = ['Date'] + self.lag_features
        self.model_feature_order = self.lag_features + self.features
        self.categorical_features = ['month', 'day_of_week', 'is_wednesday', 'is_weekend', 
                                      'is_payday_peak', 'is_payday_slump', 'is_payday_end', 'is_quarter_end']
        self.growth_map = None # Monthly YoY growth ratios
        self.inertia_params = {'intercept': 0.0, 'w_rev': 0.0, 'w_order': 1.0, 'w_aov': 0.0}
        self.q4_momentum_map = {}
        self.category_q4_momentum_map = {}
        self.q4_momentum_default = 0.0
        self.cogs_ratio_clip = (0.0, 2.0)
        
        # Scikit-Learn Pipeline for Revenue
        self.revenue_pipeline = Pipeline([
            ('features', BaselineFeatureExtractor(date_col='Date')),
            ('model', lgb.LGBMRegressor(
                n_estimators=Config.LGBM_N_ESTIMATORS,
                learning_rate=Config.LGBM_LR,
                num_leaves=Config.LGBM_NUM_LEAVES,
                objective='regression',
                random_state=42,
                verbose=-1
            ))
        ])
        
        # Scikit-Learn Pipeline for COGS Ratio
        self.cogs_pipeline = Pipeline([
            ('features', BaselineFeatureExtractor(date_col='Date')),
            ('model', lgb.LGBMRegressor(
                n_estimators=Config.LGBM_N_ESTIMATORS,
                learning_rate=Config.LGBM_LR,
                num_leaves=Config.LGBM_NUM_LEAVES,
                objective='regression',
                random_state=42,
                verbose=-1
            ))
        ])

    def _is_signaled(self, dates: pd.Series) -> pd.Series:
        """Centralized logic for event signaling (Vectorized)."""
        event_map = self.revenue_pipeline.named_steps['features'].event_score_map
        # Convert (month, day) map to (month*100 + day) for vectorized map
        event_map_vec = {m * 100 + d: True for (m, d) in event_map.keys()}
        mmdd = dates.dt.month * 100 + dates.dt.day
        is_event_discovered = mmdd.map(event_map_vec).fillna(False)
        
        return (dates.dt.day >= 25) | (dates.dt.dayofweek == 2) | is_event_discovered

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds short-term memory features using only Revenue to avoid feedback loops."""
        df = df.copy()
        # Revenue Lags are used for both models (Option A)
        df['rev_lag_1'] = df['Revenue'].shift(1)
        df['rev_lag_7'] = df['Revenue'].shift(7)
        df['rev_roll_7'] = df['Revenue'].shift(1).rolling(7, min_periods=7).mean()
        # No COGS lags to prevent noise amplification
        return df # No bfill here, handled in fit/predict

    # Removed legacy momentum methods (now in MarketAnalyst)

    def _validate_feature_contract(self, X_sample: pd.DataFrame):
        """Ensure train/inference both see the exact same feature schema."""
        transformed_rev = self.revenue_pipeline.named_steps['features'].transform(X_sample.copy())
        transformed_cogs = self.cogs_pipeline.named_steps['features'].transform(X_sample.copy())

        for model_name, transformed in [('revenue', transformed_rev), ('cogs_ratio', transformed_cogs)]:
            cols = list(transformed.columns)
            if cols != self.model_feature_order:
                raise ValueError(
                    f"Feature contract mismatch for {model_name}. "
                    f"Expected {self.model_feature_order}, got {cols}"
                )

    def _prepare_signals(self, df):
        """Discovers momentum, category events, and COGS profiles from historical data."""
        # Inject momentum into extractors
        self.q4_momentum_map, self.q4_momentum_default = MarketAnalyst.calculate_q4_momentum(df)
        self.category_q4_momentum_map = MarketAnalyst.calculate_category_q4_momentum(PROJECT_ROOT)
        
        for pipeline in [self.revenue_pipeline, self.cogs_pipeline]:
            extractor = pipeline.named_steps['features']
            extractor.set_q4_momentum_map(self.q4_momentum_map, self.q4_momentum_default)
            extractor.set_category_momentum_map(self.category_q4_momentum_map)
        
        print("Discovering Category-Specific Signals & COGS Profiles...")
        cat_event_map = MarketAnalyst.discover_category_events(df, self.revenue_pipeline.named_steps['features'].event_score_map)
        cogs_profile = df.groupby(df['Date'].dt.month).apply(
            lambda x: (x['COGS'] / (x['Revenue'] + 1e-6)).median()
        ).to_dict()
        
        for pipeline in [self.revenue_pipeline, self.cogs_pipeline]:
            extractor = pipeline.named_steps['features']
            extractor.set_category_event_map(cat_event_map)
            extractor.set_cogs_monthly_profile(cogs_profile)
            
        return cat_event_map, cogs_profile

    def _prepare_training_data(self, df):
        """Normalizes targets, adds lags, and calculates sample weights."""
        df = df.copy().sort_values('Date').reset_index(drop=True)
        df['year'] = df['Date'].dt.year

        # 1. Calculate Annual Scales for Stationarity
        self.annual_scales_rev = df.groupby('year')['Revenue'].median().to_dict()
        for yr in self.annual_scales_rev:
            if self.annual_scales_rev[yr] <= 0:
                self.annual_scales_rev[yr] = df['Revenue'].median()
        
        self.base_scale_rev = self.annual_scales_rev[sorted(self.annual_scales_rev.keys())[-1]]
        
        # 2. Add Lags and Normalize
        df_lags = self._add_lags(df[['Date', 'year', 'Revenue', 'COGS']])
        scales = df_lags['year'].map(self.annual_scales_rev).fillna(self.base_scale_rev).replace(0, 1.0)
        
        df_lags['Revenue_norm'] = df_lags['Revenue'] / scales
        df_lags['COGS_ratio'] = df_lags['COGS'] / (df_lags['Revenue'] + 1e-6)
        
        for col in self.lag_features:
            df_lags[col] = df_lags[col] / scales
            
        # 3. Capture History for Inference
        self.history_abs_rev = df_lags[['Revenue']].tail(Config.REC_HISTORY_WINDOW).copy()
        
        # 4. Weights & Cleaning
        df_lags = self._apply_data_driven_weights(df_lags)
        df_lags = df_lags.dropna().reset_index(drop=True)
        
        return df_lags

    def fit(self, df: pd.DataFrame):
        print("--- Starting Forecasting Pipeline Fit ---")
        df = df.copy().sort_values('Date').reset_index(drop=True)

        # 1. Signal Discovery
        self._prepare_signals(df)
        
        # 2. Training Data Preparation
        df_train = self._prepare_training_data(df)
        
        y_rev  = df_train['Revenue_norm']
        y_cogs = df_train['COGS_ratio']
        sample_weights = df_train['sample_weight']

        self.cogs_ratio_clip = (float(np.quantile(y_cogs, 0.01)), float(np.quantile(y_cogs, 0.99)))
        X = df_train[self.feature_cols + ['Revenue']].copy()
        
        fit_params = {
            'model__categorical_feature': self.categorical_features,
            'model__sample_weight': sample_weights
        }
        
        # 3. Model Training
        print("Training Normalized Revenue Model...")
        self.revenue_pipeline.fit(X, y_rev, **fit_params)
        
        # Sync discovered features between pipelines
        extractor = self.revenue_pipeline.named_steps['features']
        self.features = extractor.get_feature_names()
        self.model_feature_order = self.lag_features + self.features
        
        cogs_extractor = self.cogs_pipeline.named_steps['features']
        for attr in ['event_score_map', 'category_profile_map', 'categories_', 
                     'category_event_map', 'category_momentum_map', 'cogs_monthly_profile']:
            setattr(cogs_extractor, attr, getattr(extractor, attr))
        
        print("Training COGS Ratio Model...")
        self.cogs_pipeline.fit(X, y_cogs, **fit_params)

        self._validate_feature_contract(X.head(32).copy())
        
        # 4. Market Analysis & Calibration
        self.inertia_params, self.inertia_trust_weight, self.momentum_damping = MarketAnalyst.discover_inertia_params(df)
        self.momentum = MarketAnalyst.calculate_growth_calibration(
            df, extractor.event_score_map, self.inertia_params, self.inertia_trust_weight
        )
        print(f"Fit Complete. Base Momentum: {self.momentum['base']:.3f}x")
        
        return self

    # Analytical methods moved to analyst.py

    def _compute_forecast_multipliers(self, horizon_years, max_train_year):
        """Calculates compounded yearly and category-specific multipliers with damping."""
        year_multipliers = {}
        cat_multipliers = {cat: {} for cat in self.momentum['categories']}
        
        running_m_base, running_m_event = 1.0, 1.0
        cat_running_m = {cat: 1.0 for cat in self.momentum['categories']}
        
        for yr in horizon_years:
            years_out = yr - max_train_year
            # Determine damping based on horizon depth
            damp_global = 1.0 if years_out <= 1 else Config.DAMPING_Y2
            damp_cat = Config.DAMPING_Y1 if years_out <= 1 else Config.DAMPING_Y2
            
            # Global
            running_m_base *= (self.momentum['base'] ** damp_global)
            running_m_event *= (self.momentum['event'] ** damp_global)
            year_multipliers[yr] = {'base': running_m_base, 'event': running_m_event}
            
            # Categories
            for cat, mom in self.momentum['categories'].items():
                cat_running_m[cat] *= (mom ** damp_cat)
                cat_multipliers[cat][yr] = cat_running_m[cat]
                
        return year_multipliers, cat_multipliers

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Generating stationary recursive forecast (Optimized)...")
        horizon = df[['Date']].copy().sort_values('Date').reset_index(drop=True)
        horizon['year'] = horizon['Date'].dt.year
        
        # 1. Pre-calculate multipliers and signals
        max_train_year = self.momentum['max_train_year']
        horizon_years = sorted(horizon['year'].unique())
        
        year_multipliers, cat_multipliers = self._compute_forecast_multipliers(horizon_years, max_train_year)
        is_event_horizon = self._is_signaled(horizon['Date']).values
        
        # 2. Pre-transform non-lag features
        horizon_dummy = horizon.copy()
        for col in self.lag_features:
            horizon_dummy[col] = 0.0
        
        X_horizon = self.revenue_pipeline.named_steps['features'].transform(horizon_dummy)
        extractor = self.revenue_pipeline.named_steps['features']
        
        # 3. Recursive Loop
        history_rev = list(self.history_abs_rev['Revenue'].values)
        preds_rev, preds_cogs = [], []
        
        rev_model = self.revenue_pipeline.named_steps['model']
        cogs_model = self.cogs_pipeline.named_steps['model']
        
        for i in range(len(horizon)):
            curr_year = horizon.iloc[i]['year']
            
            # Calculate Blended Momentum
            blended_m = sum(
                X_horizon.iloc[i][f'share_{cat.lower()}'] * cat_multipliers[cat].get(curr_year, 1.0)
                for cat in extractor.categories_
            )
            
            # Apply Event Lift if signaled
            if is_event_horizon[i]:
                event_lift = (year_multipliers[curr_year]['event'] / (year_multipliers[curr_year]['base'] + 1e-6))
                projected_median = self.base_scale_rev * blended_m * event_lift
            else:
                projected_median = self.base_scale_rev * blended_m
            
            # Update lag features using column names for safety
            X_horizon.loc[X_horizon.index[i], 'rev_lag_1'] = history_rev[-1] / projected_median
            X_horizon.loc[X_horizon.index[i], 'rev_lag_7'] = history_rev[-7] / projected_median
            X_horizon.loc[X_horizon.index[i], 'rev_roll_7'] = (sum(history_rev[-7:]) / 7.0) / projected_median
            
            # Predict
            current_X = X_horizon.iloc[[i]]
            raw_norm_rev = float(np.clip(rev_model.predict(current_X)[0], 0.0, 5.0))
            raw_ratio = float(np.clip(cogs_model.predict(current_X)[0], self.cogs_ratio_clip[0], self.cogs_ratio_clip[1]))
            
            final_rev = max(0, raw_norm_rev * projected_median)
            preds_rev.append(final_rev)
            preds_cogs.append(max(0, final_rev * raw_ratio))
            
            history_rev.append(final_rev)
            
        return pd.DataFrame({
            'Date': horizon['Date'],
            'Revenue': preds_rev,
            'COGS':    preds_cogs
        })

    def _apply_data_driven_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        return DataWeighting.apply_weights(df, self.annual_scales_rev)

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