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
                                      'is_payday_start', 'is_payday_end', 'is_quarter_end']
        self.growth_map = None # Monthly YoY growth ratios
        self.inertia_params = {'intercept': 0.0, 'w_rev': 0.0, 'w_order': 1.0, 'w_aov': 0.0}
        self.q4_momentum_map = {}
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

    def _calculate_q4_momentum_from_raw(self, df: pd.DataFrame):
        """Compute year-level Q4 momentum from raw revenue (before normalization)."""
        tmp = df[['Date', 'Revenue']].copy()
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        tmp['year'] = tmp['Date'].dt.year
        tmp['month'] = tmp['Date'].dt.month

        q4_totals = tmp[tmp['month'] >= 10].groupby('year')['Revenue'].sum().to_dict()
        years = sorted(tmp['year'].unique())
        
        # Extend to include the momentum for the next year (e.g. 2023) if training has Q4 data
        target_years = years + [max(years) + 1]

        valid = {}
        for yr in target_years:
            prev_q4 = q4_totals.get(yr - 1)
            prev2_q4 = q4_totals.get(yr - 2)
            if prev_q4 is not None and prev2_q4 is not None and prev2_q4 > 0:
                valid[yr] = (prev_q4 / (prev2_q4 + 1e-6)) - 1.0

        default_val = float(np.median(list(valid.values()))) if valid else 0.0
        momentum_map = {yr: valid.get(yr, default_val) for yr in target_years}
        return momentum_map, default_val

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

    def fit(self, df: pd.DataFrame):
        df = df.copy().sort_values('Date').reset_index(drop=True)
        df['year'] = df['Date'].dt.year

        # 0. Pre-compute Q4 momentum from raw revenue (before any normalization)
        self.q4_momentum_map, self.q4_momentum_default = self._calculate_q4_momentum_from_raw(df)
        self.revenue_pipeline.named_steps['features'].set_q4_momentum_map(
            self.q4_momentum_map, self.q4_momentum_default
        )
        self.cogs_pipeline.named_steps['features'].set_q4_momentum_map(
            self.q4_momentum_map, self.q4_momentum_default
        )
        
        # 1. Calculate Annual Scales
        self.annual_scales_rev = df.groupby('year')['Revenue'].median().to_dict()
        
        # Stability fix: Ensure no zero scales
        for yr in self.annual_scales_rev:
            if self.annual_scales_rev[yr] <= 0:
                self.annual_scales_rev[yr] = df['Revenue'].median()
        
        years_sorted = sorted(self.annual_scales_rev.keys())
        last_year = years_sorted[-1]
        self.base_scale_rev = self.annual_scales_rev[last_year]
        
        # 2. Add Lags on ABSOLUTE values first to ensure continuity
        df_lags = self._add_lags(df[['Date', 'year', 'Revenue', 'COGS']])
        
        # 3. Normalize Targets & Lags by the current year's scale (Stationary Transformation)
        scales = df_lags['year'].map(self.annual_scales_rev).fillna(self.base_scale_rev).replace(0, 1.0)
        
        df_lags['Revenue_norm'] = df_lags['Revenue'] / scales
        df_lags['COGS_ratio'] = df_lags['COGS'] / (df_lags['Revenue'] + 1e-6)
        
        for col in self.lag_features:
            df_lags[col] = df_lags[col] / scales
            
        # 4. Prepare History for Inference (Absolute values)
        self.history_abs_rev = df_lags[['Revenue']].tail(Config.REC_HISTORY_WINDOW).copy()
        
        # 5. Apply Data-driven Weights
        df_lags = self._apply_data_driven_weights(df_lags)
        
        # Clean up NaNs from lags/rolling
        df_lags = df_lags.dropna().reset_index(drop=True)
        
        y_rev  = df_lags['Revenue_norm']
        y_cogs = df_lags['COGS_ratio']
        sample_weights = df_lags['sample_weight']

        ratio_q01 = float(np.quantile(y_cogs, 0.01))
        ratio_q99 = float(np.quantile(y_cogs, 0.99))
        self.cogs_ratio_clip = (ratio_q01, ratio_q99)
        
        # Feature columns (Include Revenue for extractor discovery, it will be dropped in transform)
        X = df_lags[self.feature_cols + ['Revenue']].copy()
        
        fit_params = {
            'model__categorical_feature': self.categorical_features,
            'model__sample_weight': sample_weights
        }
        
        print("Training Normalized Revenue Model...")
        self.revenue_pipeline.fit(X, y_rev, **fit_params)
        
        # Update feature contract based on dynamic discovery from the fitted extractor
        extractor = self.revenue_pipeline.named_steps['features']
        self.features = extractor.get_feature_names()
        self.model_feature_order = self.lag_features + self.features
        
        # Share discovered signals with COGS model to avoid redundant computation
        self.cogs_pipeline.named_steps['features'].event_score_map = extractor.event_score_map
        self.cogs_pipeline.named_steps['features'].category_profile_map = extractor.category_profile_map
        self.cogs_pipeline.named_steps['features'].categories_ = extractor.categories_
        
        print("Training COGS Ratio Model...")
        self.cogs_pipeline.fit(X, y_cogs, **fit_params)

        self._validate_feature_contract(X.head(32).copy())

        # 3. Growth Calibration & Weight Discovery
        self._discover_inertia_params(df)
        self._calculate_growth_calibration(df)
        
        return self

    def _discover_inertia_params(self, df: pd.DataFrame):
        """
        Learns the relationship between Q4 signals and next-year growth using historical data.
        Rule: No magic numbers. Weights are derived via Log-Linear Regression.
        """
        print("Starting Dynamic Inertia Weight Discovery...")
        
        # 1. Prepare historical yearly stats
        tmp = df.copy()
        tmp['year'] = tmp['Date'].dt.year
        tmp['month'] = tmp['Date'].dt.month
        
        annual_medians = tmp.groupby('year')['Revenue'].median().to_dict()
        
        # Aggregate Q4 signals
        q4_data = tmp[tmp['month'] >= 10].copy()
        q4_rev = q4_data.groupby('year')['Revenue'].sum()
        
        orders_full = pd.read_parquet(Config.ORDERS_FILE)
        orders_full['order_date'] = pd.to_datetime(orders_full['order_date'])
        
        max_date = df['Date'].max()
        orders_full = orders_full[orders_full['order_date'] <= max_date]
        
        q4_orders = orders_full[orders_full['order_date'].dt.month >= 10].groupby(orders_full['order_date'].dt.year)['order_id'].count()
        
        avail_years = sorted(list(set(q4_rev.index) & set(q4_orders.index) & set(annual_medians.keys())))
        
        if len(avail_years) < 3:
            print("Warning: Insufficient historical years for regression. Using default Order-priority weights.")
            return

        # 2. Build training set for Inertia Model
        rows = []
        for i in range(2, len(avail_years)):
            yr_target = avail_years[i]
            yr_prev = avail_years[i-1]
            yr_prev2 = avail_years[i-2]
            
            # Target: Log of realized annual growth
            log_g = np.log(annual_medians[yr_target] / (annual_medians[yr_prev] + 1e-6) + 1e-6)
            
            # Signals: Log of Q4 YoY Momentum
            log_m_rev = np.log(q4_rev[yr_prev] / (q4_rev[yr_prev2] + 1e-6) + 1e-6)
            log_m_order = np.log(q4_orders[yr_prev] / (q4_orders[yr_prev2] + 1e-6) + 1e-6)
            log_m_aov = log_m_rev - log_m_order # log(A/B) = log(A) - log(B)
            
            rows.append({'y': log_g, 'x_rev': log_m_rev, 'x_order': log_m_order, 'x_aov': log_m_aov})
            
        train_df = pd.DataFrame(rows)
        
        # 3. Simple Linear Regression to find Beta weights
        # We use a basic OLS approach via numpy for minimal overhead
        X = train_df[['x_rev', 'x_order', 'x_aov']].values
        X = np.hstack([np.ones((X.shape[0], 1)), X]) # Add intercept
        y = train_df['y'].values
        
        try:
            # Solve (X^T X)^-1 X^T y
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            # Calculate R-squared for trust weighting
            # SST = sum((y - y_mean)^2), SSR = sum(residuals)
            y_mean = np.mean(y)
            sst = np.sum((y - y_mean)**2)
            ssr = residuals[0] if len(residuals) > 0 else 0.0
            r_squared = 1 - (ssr / (sst + 1e-6))
            
            self.inertia_params = {
                'intercept': float(coeffs[0]),
                'w_rev':     float(coeffs[1]),
                'w_order':   float(coeffs[2]),
                'w_aov':     float(coeffs[3])
            }
            
            # Trust inertia more if R-squared is high, but floor it at 0.5 to avoid neglecting structural signals
            self.inertia_trust_weight = np.clip(r_squared, 0.5, 0.95)
            
            # Also derive MOMENTUM_DAMPING from YoY volatility
            # High volatility -> more damping (lower damping factor)
            yoy_volatility = np.std(np.exp(y))
            self.momentum_damping = np.clip(1.0 - (yoy_volatility * 0.5), 0.7, 0.98)
            
            print(f"Learned Inertia Weights: Rev={self.inertia_params['w_rev']:.3f}, Order={self.inertia_params['w_order']:.3f}, AOV={self.inertia_params['w_aov']:.3f}")
            print(f"Inertia Confidence (R2): {r_squared:.3f} -> Trust Weight: {self.inertia_trust_weight:.2f}")
            print(f"Data-driven Damping: {self.momentum_damping:.3f} (based on YoY volatility {yoy_volatility:.3f})")
            
        except Exception as e:
            print(f"Regression failed: {e}. Falling back to default heuristics.")
            self.inertia_trust_weight = 0.8
            self.momentum_damping = 0.9

    def _calculate_growth_calibration(self, df: pd.DataFrame):
        """
        Calculates Dual-Momentum YoY growth: Base Growth vs Event Momentum.
        Rule 10: Derived dynamically from training data.
        """
        # 1. Setup Data & Signaling
        max_date = df['Date'].max()
        traffic = pd.read_parquet(Config.WEB_TRAFFIC_FILE)
        traffic['date'] = pd.to_datetime(traffic['date'])
        traffic = traffic[traffic['date'] <= max_date]
        
        orders = pd.read_parquet(Config.ORDERS_FILE)
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        orders = orders[orders['order_date'] <= max_date]
        
        daily_traffic = traffic.groupby('date')['sessions'].sum()
        daily_orders = orders.groupby('order_date')['order_id'].count()
        daily_rev = df.groupby('Date')['Revenue'].sum()
        
        def is_signaled_local(dates):
            # Paydays End (25-31), Wednesdays, or Discovered Events
            event_map = self.revenue_pipeline.named_steps['features'].event_score_map
            return (dates.day >= 25) | (dates.dayofweek == 2) | \
                   (dates.map(lambda x: (x.month, x.day) in event_map))

        # 2. Dynamic Momentum Window Discovery
        window_results = []
        for w in Config.MOMENTUM_WINDOWS:
            curr_start = max_date - pd.Timedelta(days=w)
            ref_start, ref_end = curr_start - pd.DateOffset(years=1), max_date - pd.DateOffset(years=1)
            
            if ref_start < daily_traffic.index.min(): continue
            
            # Helper to calculate Bi-Factor Lift (Non-Circular: Traffic + Orders + AOV)
            def get_lift(mask_name):
                lifts = []
                for factor_data in [daily_orders, daily_rev / (daily_orders + 1e-6)]:
                    curr_vals = factor_data[(factor_data.index > curr_start) & (factor_data.index <= max_date)]
                    ref_vals = factor_data[(factor_data.index > ref_start) & (factor_data.index <= ref_end)]
                    
                    if mask_name == 'event':
                        curr_vals = curr_vals[is_signaled_local(curr_vals.index)]
                        ref_vals = ref_vals[is_signaled_local(ref_vals.index)]
                    
                    c_mean, r_mean = curr_vals.mean(), ref_vals.mean()
                    lifts.append(c_mean / (r_mean + 1e-6) if not (np.isnan(c_mean) or np.isnan(r_mean)) else 1.0)
                
                return np.clip(np.prod(lifts), Config.MOMENTUM_CLIP_MIN, Config.MOMENTUM_CLIP_MAX), np.std(lifts)

            base_g, base_std = get_lift('all')
            window_results.append({'window': w, 'base_g': base_g, 'cv': base_std / (base_g + 1e-6)})

        # 3. Final Selection & Momentum calculation
        if not window_results:
            print("Warning: No valid momentum windows found. Using default 1.0x.")
            self.momentum = {'base': 1.0, 'event': 1.0, 'max_train_year': max_date.year}
            return

        best_res = min(window_results, key=lambda x: x['cv'])
        w = best_res['window']
        
        # Ensure we use the best window parameters for final multipliers
        curr_start = max_date - pd.Timedelta(days=w)
        ref_start, ref_end = curr_start - pd.DateOffset(years=1), max_date - pd.DateOffset(years=1)

        def get_direct_rev_momentum(start, end, mode='all'):
            stats = {}
            for p_name, (p_start, p_end) in {'curr': (start, end), 'ref': (ref_start, ref_end)}.items():
                def get_weighted_rev(series, start_d, end_d):
                    full_range = pd.date_range(start=start_d + pd.Timedelta(days=1), end=end_d)
                    if mode == 'event': full_range = full_range[is_signaled_local(full_range)]
                    elif mode == 'base': full_range = full_range[~is_signaled_local(full_range)]
                    
                    if full_range.empty: return 0.0, 1e-6
                    
                    days_diff = (end_d - full_range).days
                    weights = np.exp(-days_diff / float(Config.MOMENTUM_DECAY_DAYS))
                    total_w = np.sum(weights)
                    
                    weighted_data = series.reindex(full_range, fill_value=0.0) * weights
                    return np.sum(weighted_data), total_w
                
                sum_r, w_r = get_weighted_rev(daily_rev, p_start, p_end)
                stats[p_name] = sum_r / (w_r + 1e-6)
            
            lift = stats['curr'] / (stats['ref'] + 1e-6)
            return np.clip(lift, Config.MOMENTUM_CLIP_MIN, Config.MOMENTUM_CLIP_MAX)

        # --- Dynamic Multi-Factor Inertia Calculation ---
        # 1. Get latest complete Q4 signals
        q4_data = df[df['Date'].dt.month >= 10].copy()
        q4_rev_sum = q4_data.groupby(q4_data['Date'].dt.year)['Revenue'].sum()
        
        orders_full = pd.read_parquet(Config.ORDERS_FILE)
        orders_full['order_date'] = pd.to_datetime(orders_full['order_date'])
        q4_orders_sum = orders_full[orders_full['order_date'].dt.month >= 10].groupby(orders_full['order_date'].dt.year)['order_id'].count()
        
        avail_years = sorted(list(set(q4_rev_sum.index) & set(q4_orders_sum.index)))
        last_yr = avail_years[-1]
        prev_yr = avail_years[-2]
        
        log_m_rev = np.log(q4_rev_sum[last_yr] / (q4_rev_sum[prev_yr] + 1e-6) + 1e-6)
        log_m_order = np.log(q4_orders_sum[last_yr] / (q4_orders_sum[prev_yr] + 1e-6) + 1e-6)
        log_m_aov = log_m_rev - log_m_order
        
        # 2. Apply Learned Regression Model: M = exp(intercept + sum(w_i * log_m_i))
        p = self.inertia_params
        log_calibrated = p['intercept'] + (p['w_rev'] * log_m_rev) + (p['w_order'] * log_m_order) + (p['w_aov'] * log_m_aov)
        calibrated_m = np.exp(log_calibrated)
        
        # Stability Damping
        calibrated_m = np.clip(calibrated_m, 0.7, 1.8)
        
        self.momentum = {}
        raw_base_m = get_direct_rev_momentum(curr_start, max_date, 'base')
        raw_event_m = get_direct_rev_momentum(curr_start, max_date, 'event')
        
        # Final Blend: Weight based on dynamically discovered trust factor
        i_w = self.inertia_trust_weight
        self.momentum['base']  = np.clip(raw_base_m * (1 - i_w) + calibrated_m * i_w, Config.MOMENTUM_CLIP_MIN, Config.MOMENTUM_CLIP_MAX)
        self.momentum['event'] = np.clip(raw_event_m * (1 - i_w) + calibrated_m * i_w, Config.MOMENTUM_CLIP_MIN, Config.MOMENTUM_CLIP_MAX)
        self.momentum['max_train_year'] = max_date.year
        
        print(f"Dynamic Inertia Applied (Calibrated M: {calibrated_m:.3f}x)")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Generating stationary recursive forecast (Optimized)...")
        horizon = df[['Date']].copy().sort_values('Date').reset_index(drop=True)
        
        # Pre-calculate compounded multipliers and event signals for the entire horizon
        max_train_year = self.momentum['max_train_year']
        base_m = self.momentum['base']
        event_m = self.momentum['event']
        
        horizon['year'] = horizon['Date'].dt.year
        horizon_years = sorted(horizon['year'].unique())
        year_multipliers = {}
        running_m_base, running_m_event = 1.0, 1.0
        
        for yr in horizon_years:
            years_out = yr - max_train_year
            damp = self.momentum_damping ** max(0, years_out - 1)
            running_m_base *= (base_m ** damp)
            running_m_event *= (event_m ** damp)
            year_multipliers[yr] = {'base': running_m_base, 'event': running_m_event}

        # Pre-calculate event signals for the whole horizon
        is_event_horizon = self._is_signaled(horizon['Date']).values
        
        # Pre-transform non-lag features using a dummy DataFrame to satisfy the transformer
        horizon_dummy = horizon.copy()
        for col in self.lag_features:
            horizon_dummy[col] = 0.0
        
        # X_horizon will contain all features in the correct order
        X_horizon = self.revenue_pipeline.named_steps['features'].transform(horizon_dummy)
        
        # Use lists for faster buffer management (Storing ABSOLUTE values)
        history_rev = list(self.history_abs_rev['Revenue'].values)
        
        preds_rev = []
        preds_cogs = []
        
        # Cache model references
        rev_model = self.revenue_pipeline.named_steps['model']
        cogs_model = self.cogs_pipeline.named_steps['model']
        
        # Recursive Loop
        for i in range(len(horizon)):
            curr_year = horizon.iloc[i]['year']
            is_event = is_event_horizon[i]
            effective_m = year_multipliers[curr_year]['event' if is_event else 'base']
            projected_median = self.base_scale_rev * effective_m
            
            # 1. Update lag features (Normalized on-the-fly to the current projected scale)
            lag_1 = history_rev[-1] / projected_median
            lag_7 = history_rev[-7] / projected_median
            roll_7 = (sum(history_rev[-7:]) / 7.0) / projected_median
            
            X_horizon.iloc[i, 0] = lag_1
            X_horizon.iloc[i, 1] = lag_7
            X_horizon.iloc[i, 2] = roll_7
            
            # 2. Model Prediction (Outputs Normalized Shape)
            current_X = X_horizon.iloc[[i]]
            raw_norm_rev = float(np.clip(rev_model.predict(current_X)[0], 0.0, 5.0))
            raw_ratio = float(np.clip(cogs_model.predict(current_X)[0], self.cogs_ratio_clip[0], self.cogs_ratio_clip[1]))
            
            # 3. Scaling back to Absolute
            final_rev = max(0, raw_norm_rev * projected_median)
            final_cogs = max(0, final_rev * raw_ratio)
            
            preds_rev.append(final_rev)
            preds_cogs.append(final_cogs)
            
            # 4. Update Buffer with ABSOLUTE Revenue
            history_rev.append(final_rev)
            
        return pd.DataFrame({
            'Date': horizon['Date'],
            'Revenue': preds_rev,
            'COGS':    preds_cogs
        })

    def _apply_data_driven_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates weights for each sample based on:
        1. Recency (Trust recent data more)
        2. Efficiency Consistency (Discount anomalies in Rev/Sessions ratio relative to yearly median)
        """
        print("Calculating Data-driven Sample Weights...")
        df = df.copy()
        
        # A. Recency Weight
        start_date_ref = df['Date'].min()
        days_from_start = (df['Date'] - start_date_ref).dt.days
        max_days = days_from_start.max()
        recency_weight = np.exp((days_from_start - max_days) / (365.0 * Config.DECAY_HALF_LIFE_YEARS))
        
        # B. Anomaly Weight (Efficiency Ratio context)
        # 1. Load Traffic for alignment
        traffic = pd.read_parquet(Config.WEB_TRAFFIC_FILE)
        traffic['date'] = pd.to_datetime(traffic['date'])
        daily_traffic = traffic.groupby('date')['sessions'].sum()
        
        # 2. Monthly Stats for Z-Score
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['sessions'] = df['Date'].map(daily_traffic).fillna(1e-6)
        
        # We need Revenue to compute the ratio (y is available in df at this point of fit)
        # However, for training we use the target y. Let's use the actual Revenue before normalization if possible.
        # But wait, df here is df_lags which has normalized Revenue.
        # Let's map back to annual scale to get raw revenue for the ratio.
        df['raw_rev'] = df['Revenue'] * df['year'].map(self.annual_scales_rev)
        
        monthly_stats = df.groupby(['year', 'month']).agg({
            'raw_rev': 'sum',
            'sessions': 'sum'
        }).reset_index()
        monthly_stats['eff_ratio'] = monthly_stats['raw_rev'] / (monthly_stats['sessions'] + 1e-6)
        
        # Yearly Median/Std for Ratio
        yearly_meta = monthly_stats.groupby('year')['eff_ratio'].agg(['median', 'std']).reset_index()
        yearly_meta.columns = ['year', 'y_median', 'y_std']
        
        monthly_stats = pd.merge(monthly_stats, yearly_meta, on='year')
        monthly_stats['z_score'] = (monthly_stats['eff_ratio'] - monthly_stats['y_median']).abs() / (monthly_stats['y_std'] + 1e-6)
        monthly_stats['outlier_weight'] = 1.0 / (1.0 + monthly_stats['z_score'])
        
        # 3. Map weights back to daily rows (Vectorized)
        weight_map = monthly_stats.set_index(['year', 'month'])['outlier_weight'].to_dict()
        
        # Using a MultiIndex map is significantly faster than .apply(axis=1)
        df_index = pd.MultiIndex.from_arrays([df['year'], df['month']])
        outlier_weights = df_index.map(weight_map).fillna(1.0)
        
        # Combine
        df['sample_weight'] = recency_weight * outlier_weights
        
        # Clean up temp columns
        return df.drop(columns=['raw_rev', 'sessions', 'year', 'month'])

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
