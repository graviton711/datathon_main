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
        self.features = ['month', 'day', 'day_of_week', 'is_weekend', 'days_to_tet', 'event_score', 'prev_q4_momentum']
        self.categorical_features = ['month', 'day_of_week', 'is_weekend']
        self.growth_map = None # Monthly YoY growth ratios
        
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

    def fit(self, df: pd.DataFrame):
        print("Pre-processing outliers in training data...")
        df = self._smooth_training_outliers(df)
        
        print("Training Baseline Revenue Model...")
        # X chỉ chứa Date, y chứa Revenue/Target
        X = df[['Date']].copy()
        y_rev = df['Revenue']
        y_cogs_ratio = df['COGS'] / (df['Revenue'] + 1e-6)
        
        # Calculate time-decay weights inside fit
        start_date_ref = df['Date'].min()
        days_from_start = (df['Date'] - start_date_ref).dt.days
        max_days = days_from_start.max()
        # Memory Tuning: Relax decay to preserve 10yr historical signals (Rule 10)
        sample_weights = np.exp((days_from_start - max_days) / (365.0 * 2.5))
        
        # Train pipelines
        # LightGBM categorical features must be passed in fit_params
        fit_params = {
            'model__categorical_feature': self.categorical_features,
            'model__sample_weight': sample_weights
        }
        
        self.revenue_pipeline.fit(X, y_rev, **fit_params)
        
        print("Training Baseline COGS Ratio Model...")
        self.cogs_pipeline.fit(X, y_cogs_ratio, **fit_params)

        # Calculate Data-Driven Growth Calibration Map (Rule 10 compliant)
        self._calculate_growth_calibration(df)
        
        return self

    def _calculate_growth_calibration(self, df: pd.DataFrame):
        """
        Calculates Dual-Momentum YoY growth: Base Growth vs Event Momentum.
        Rule 10: Derived dynamically from training data.
        """
        # 1. Setup Data & Signaling
        max_date = df['Date'].max()
        traffic = pd.read_parquet('data/processed/web_traffic.parquet')
        traffic['date'] = pd.to_datetime(traffic['date'])
        traffic = traffic[traffic['date'] <= max_date]
        
        orders = pd.read_parquet('data/processed/orders.parquet')
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        orders = orders[orders['order_date'] <= max_date]
        
        daily_traffic = traffic.groupby('date')['sessions'].sum()
        daily_orders = orders.groupby('order_date')['order_id'].count()
        daily_rev = df.groupby('Date')['Revenue'].sum()
        
        # Get discovered signals from the feature extractor
        event_map = self.revenue_pipeline.named_steps['features'].event_score_map
        
        def is_signaled(dates):
            # Paydays (25-5) or Discovered Events
            return (dates.day >= 25) | (dates.day <= 5) | \
                   (dates.map(lambda x: (x.month, x.day) in event_map))

        # 2. Dynamic Momentum Window Discovery (using All Days for stability)
        candidate_windows = [90, 180, 270, 365]
        window_results = []
        
        for w in candidate_windows:
            curr_start = max_date - pd.Timedelta(days=w)
            ref_start, ref_end = curr_start - pd.DateOffset(years=1), max_date - pd.DateOffset(years=1)
            
            if ref_start < daily_traffic.index.min(): continue
            
            # Helper to calculate tri-factor lift for a mask
            def get_lift(mask_name):
                lifts = []
                for factor_data, date_col in [(daily_traffic, 'index'), (daily_orders, 'index'), (daily_rev, 'index')]:
                    curr_vals = factor_data[(factor_data.index > curr_start) & (factor_data.index <= max_date)]
                    ref_vals = factor_data[(factor_data.index > ref_start) & (factor_data.index <= ref_end)]
                    
                    # Apply specific mask (e.g. all days, or just event days)
                    if mask_name == 'event':
                        curr_vals = curr_vals[is_signaled(curr_vals.index)]
                        ref_vals = ref_vals[is_signaled(ref_vals.index)]
                    
                    c_mean, r_mean = curr_vals.mean(), ref_vals.mean()
                    lifts.append(c_mean / (r_mean + 1e-6) if not (np.isnan(c_mean) or np.isnan(r_mean)) else 1.0)
                
                # Combined Growth: Sessions * (Orders/Sessions) * (Rev/Orders)
                # This simplifies to Rev_curr / Rev_ref
                return np.clip(np.prod(lifts), 0.5, 2.5), np.std(lifts)

            base_g, base_std = get_lift('all')
            window_results.append({'window': w, 'base_g': base_g, 'cv': base_std / (base_g + 1e-6)})

        # 3. Final Selection & Momentum calculation (Recent-Weighted & Tightly Separated)
        best_res = min(window_results, key=lambda x: x['cv'])
        w = best_res['window']
        
        def get_weighted_tri_factor(start, end, mode='all'):
            # Standardized Tri-Factor Components (Aligned per series)
            stats = {}
            for p_name, (p_start, p_end) in {'curr': (start, end), 'ref': (ref_start, ref_end)}.items():
                def get_aligned_avg(series, start_d, end_d):
                    if series.empty: return 1e-6
                    m = (series.index > start_d) & (series.index <= end_d)
                    if mode == 'event': m &= is_signaled(series.index)
                    elif mode == 'base': m &= ~is_signaled(series.index)
                    
                    subset = series[m]
                    if subset.empty: return 1e-6
                    
                    # Helper for weighted average (Exponential decay towards end of period)
                    days_diff = (end_d - subset.index).days
                    weights = np.exp(-days_diff / 120.0)
                    return np.average(subset, weights=weights)
                
                avg_s = get_aligned_avg(daily_traffic, p_start, p_end)
                avg_o = get_aligned_avg(daily_orders, p_start, p_end)
                avg_r = get_aligned_avg(daily_rev, p_start, p_end)
                
                stats[p_name] = {
                    'sessions': avg_s,
                    'cr': avg_o / (avg_s + 1e-6),
                    'aov': avg_r / (avg_o + 1e-6)
                }
            
            # Calculate final compounding lift
            t_lift = stats['curr']['sessions'] / stats['ref']['sessions']
            cr_lift = stats['curr']['cr'] / stats['ref']['cr']
            aov_lift = stats['curr']['aov'] / stats['ref']['aov']
            
            return np.clip(t_lift * cr_lift * aov_lift, 0.5, 2.5), f"T={t_lift:.2f}, CR={cr_lift:.2f}, AOV={aov_lift:.2f}"

        # 4. Store Multipliers
        self.momentum = {}
        self.momentum['base'], base_audit = get_weighted_tri_factor(curr_start, max_date, 'base')
        self.momentum['event'], event_audit = get_weighted_tri_factor(curr_start, max_date, 'event')
        self.momentum['max_train_year'] = max_date.year
        
        # 5. Monthly Seasonality Bias
        m_curr = df[df['Date'].dt.year == max_date.year].groupby(df['Date'].dt.month)['Revenue'].mean()
        m_prev = df[df['Date'].dt.year == (max_date.year-1)].groupby(df['Date'].dt.month)['Revenue'].mean()
        self.seasonal_bias = (m_curr / (m_prev.replace(0, np.nan) * self.momentum['base'])).fillna(1.0).to_dict()
        
        print(f"Momentum Discovered (Weighted Window: {w}d):")
        print(f" > Base  (Normal): {self.momentum['base']:.3f}x ({base_audit})")
        print(f" > Event (Signaled): {self.momentum['event']:.3f}x ({event_audit})")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Generating forecast...")
        X = df[['Date']].copy()
        
        preds_rev = self.revenue_pipeline.predict(X)
        preds_ratio = self.cogs_pipeline.predict(X)
        
        # Apply Dual-Momentum Calibration
        if hasattr(self, 'momentum'):
            years_out = X['Date'].dt.year - self.momentum['max_train_year']
            months = X['Date'].dt.month
            days = X['Date'].dt.day
            
            # Identify signaled days in prediction set
            event_map = self.revenue_pipeline.named_steps['features'].event_score_map
            is_event = (days >= 25) | (days <= 5) | (X['Date'].map(lambda x: (x.month, x.day) in event_map))
            
            # Apply multipliers
            base_mult = self.momentum['base'] ** years_out
            event_mult = self.momentum['event'] ** years_out
            
            final_mult = np.where(is_event, event_mult, base_mult)
            
            # Apply Monthly Bias & Growth
            bias = X['Date'].dt.month.map(self.seasonal_bias).fillna(1.0)
            preds_rev = preds_rev * final_mult * bias
            
        return pd.DataFrame({
            'Date': df['Date'],
            'Revenue': np.maximum(0, preds_rev),
            'COGS': np.maximum(0, preds_rev * preds_ratio)
        })

    def _smooth_training_outliers(self, df: pd.DataFrame):
        """
        Detects and heals monthly outliers based on YoY growth ratios.
        This handles events like the 2021 lockdown by normalizing them to the growth trend.
        """
        df = df.copy()
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        
        # 1. Calculate Monthly Revenue
        monthly_rev = df.groupby(['year', 'month'])['Revenue'].sum().reset_index()
        monthly_rev = monthly_rev.sort_values(['year', 'month'])
        
        # 2. Calculate YoY Growth Ratios
        monthly_rev['prev_year_rev'] = monthly_rev.groupby('month')['Revenue'].shift(1)
        monthly_rev['yoy_ratio'] = monthly_rev['Revenue'] / (monthly_rev['prev_year_rev'] + 1e-6)
        
        # 3. Identify Outliers (Ratio < 0.6 or > 1.8)
        # We also calculate the median ratio per year to see the 'Normal' growth
        median_ratio = monthly_rev['yoy_ratio'].median()
        is_outlier = (monthly_rev['yoy_ratio'] < 0.6) | (monthly_rev['yoy_ratio'] > 1.8)
        
        outliers = monthly_rev[is_outlier].copy()
        if not outliers.empty:
            print(f"Detected {len(outliers)} monthly outliers. Smoothing to median growth ({median_ratio:.2f}x)...")
            for idx, row in outliers.iterrows():
                yr, mt = row['year'], row['month']
                target_rev = row['prev_year_rev'] * median_ratio
                correction_factor = target_rev / (row['Revenue'] + 1e-6)
                
                # Apply correction factor to daily data for this specific month
                mask = (df['year'] == yr) & (df['month'] == mt)
                df.loc[mask, 'Revenue'] *= correction_factor
        
        return df

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
