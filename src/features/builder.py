import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import Config
from src.training.analyst import MarketAnalyst

class BaselineFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A modular Scikit-Learn compatible transformer for baseline feature extraction.
    Logic is split into specialized sub-methods for maintainability.
    """
    def __init__(self, date_col='Date', rev_col='Revenue'):
        self.date_col = date_col
        self.rev_col = rev_col
        self.start_date_ref = None
        self.q4_momentum_dict = {}
        self.q4_momentum_default = 0.0
        self.event_score_map = {} 
        self.category_event_map = {} 
        self.category_momentum_map = {} 
        self.category_profile_map = {} 
        self.cogs_monthly_profile = {} 
        self.categories_ = [] 
        self.latest_peak_lift = 1.0 
        
        self.tet_dates = {}  # populated in fit() from data via MarketAnalyst._infer_tet_dates

    def set_q4_momentum_map(self, q4_momentum_dict, default_value=None):
        self.q4_momentum_dict = {int(k): float(v) for k, v in q4_momentum_dict.items()} if q4_momentum_dict else {}
        self.q4_momentum_default = float(default_value) if default_value is not None else (float(np.median(list(self.q4_momentum_dict.values()))) if self.q4_momentum_dict else 0.0)
        return self

    def set_cogs_monthly_profile(self, profile_map):
        self.cogs_monthly_profile = {int(k): float(v) for k, v in profile_map.items()}
        return self

    def set_category_event_map(self, category_event_map):
        self.category_event_map = category_event_map
        return self

    def set_category_momentum_map(self, category_momentum_map):
        self.category_momentum_map = category_momentum_map
        return self

    def fit(self, X, y=None):
        if y is not None:
            df = X.copy()
            df[self.rev_col] = y
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            self.start_date_ref = df[self.date_col].min()

            max_train_year = df[self.date_col].dt.year.max()
            train_years = df[self.date_col].dt.year.unique().tolist()
            extra_years = list(range(max_train_year + 1, max_train_year + 3))
            self.tet_dates = MarketAnalyst._infer_tet_dates(train_years + extra_years)

            if not self.event_score_map:
                self.event_score_map = MarketAnalyst.discover_global_events(df)
            if not self.category_profile_map:
                self.category_profile_map, self.categories_ = MarketAnalyst.discover_category_profiles(df[self.date_col].max())
            self.latest_peak_lift = MarketAnalyst.discover_peak_momentum(df)
        return self

    def transform(self, X):
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            
        # Modular Feature Construction
        self._add_time_features(X)
        self._add_seasonal_signals(X)
        self._add_event_signals(X)
        self._add_category_mix(X)
        self._add_momentum_signals(X)
        self._add_cyclic_features(X)
            
        # Final cleanup and ordering
        X = X.drop(columns=[self.date_col, 'mmdd', 'year'], errors='ignore')
        if self.rev_col in X.columns: X = X.drop(columns=[self.rev_col])
        
        ordered_cols = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7'] + self.get_feature_names()
        return X[ordered_cols] if all(col in X.columns for col in ordered_cols) else X

    def _add_time_features(self, X):
        X['year'] = X[self.date_col].dt.year
        X['month'] = X[self.date_col].dt.month
        X['day'] = X[self.date_col].dt.day
        X['day_of_week'] = X[self.date_col].dt.dayofweek
        X['is_wednesday'] = (X['day_of_week'] == 2).astype(int)
        X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)

    def _add_seasonal_signals(self, X):
        X['is_payday_peak'] = ((X['day'] >= 1) & (X['day'] <= 2)).astype(int)
        X['is_payday_slump'] = ((X['day'] >= 3) & (X['day'] <= 6)).astype(int)
        X['is_payday_end'] = (X['day'] >= 27).astype(int)
        X['is_quarter_end'] = (X['month'].isin([3, 6, 9, 12]) & (X['day'] >= 24)).astype(int)
        X['is_odd_year_aug'] = ((X['year'] % 2 != 0) & (X['month'] == 8)).astype(int)
        X['days_to_tet'] = self._vectorize_days_to_tet(X[self.date_col])
        
        # Double Day Sales (9/9, 10/10, 11/11, 12/12)
        X['is_double_day'] = ((X['month'] == X['day']) & (X['month'] >= 9)).astype(int)

    def _add_event_signals(self, X):
        X['mmdd'] = X['month'] * 100 + X['day']
        event_map_vec = {m * 100 + d: score for (m, d), score in self.event_score_map.items()}
        X['event_score'] = X['mmdd'].map(event_map_vec).fillna(1.0)
        
        for cat in self.categories_:
            cat_l = cat.lower()
            cat_ev_map = self.category_event_map.get(cat, {})
            cat_ev_vec = {m * 100 + d: score for (m, d), score in cat_ev_map.items()}
            X[f'event_score_{cat_l}'] = X['mmdd'].map(cat_ev_vec).fillna(1.0)

    def _add_category_mix(self, X):
        for cat in self.categories_:
            cat_l = cat.lower()
            cat_map = {month: shares.get(cat, 0.0) for month, shares in self.category_profile_map.items()}
            X[f'share_{cat_l}'] = X['month'].map(cat_map).fillna(0.0)
            X[f'inter_{cat_l}'] = X['event_score'] * X[f'share_{cat_l}']
        X['cogs_profile'] = X['month'].map(self.cogs_monthly_profile).fillna(0.85)

    def _add_momentum_signals(self, X):
        # Global Q4 Momentum
        unique_years = X['year'].unique()
        y_mom_map = {y: self._resolve_prev_q4_momentum(y) for y in unique_years}
        X['prev_q4_momentum'] = X['year'].map(y_mom_map)
        
        # Category-Specific Blended Momentum
        X['cat_blended_momentum'] = 0.0
        for y in unique_years:
            mask = X['year'] == y
            if mask.any():
                cat_mom_year = self.category_momentum_map.get(y, {})
                if not cat_mom_year:
                    X.loc[mask, 'cat_blended_momentum'] = X.loc[mask, 'prev_q4_momentum']
                else:
                    blended = pd.Series(0.0, index=X[mask].index)
                    for cat in self.categories_:
                        blended += X.loc[mask, f'share_{cat.lower()}'] * cat_mom_year.get(cat, self.q4_momentum_default)
                    X.loc[mask, 'cat_blended_momentum'] = blended

        # Peak Momentum Signal
        if self.rev_col in X.columns:
            m_baseline = X.groupby(['year', 'month'])[self.rev_col].transform('mean')
            lift_tmp = X[self.rev_col] / (m_baseline + 1e-6)
            peak_val_tmp = np.where(lift_tmp > 2.0, lift_tmp, np.nan)
            X['peak_momentum'] = X['event_score'] * pd.Series(peak_val_tmp).shift(1).ffill().fillna(1.0).values
        else:
            X['peak_momentum'] = X['event_score'] * self.latest_peak_lift

    def _add_cyclic_features(self, X):
        dim = X[self.date_col].dt.days_in_month
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / dim)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / dim)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

    def _vectorize_days_to_tet(self, dates):
        tet_dates = np.array(list(self.tet_dates.values()), dtype='datetime64[ns]')
        dates_np = dates.values.astype('datetime64[ns]')[:, np.newaxis]
        diffs = (dates_np - tet_dates).astype('timedelta64[D]').astype(int)
        min_idx = np.argmin(np.abs(diffs), axis=1)
        closest_diff = diffs[np.arange(len(dates)), min_idx]
        return np.where((closest_diff >= -60) & (closest_diff <= 20), closest_diff, 999)

    def _resolve_prev_q4_momentum(self, year):
        if year in self.q4_momentum_dict: return self.q4_momentum_dict[year]
        known_years = [y for y in self.q4_momentum_dict if y < year]
        return self.q4_momentum_dict[max(known_years)] if known_years else self.q4_momentum_default

    def get_feature_names(self):
        base = ['month', 'day', 'day_of_week', 'is_wednesday', 'is_weekend', 'is_payday_peak', 'is_payday_slump', 'is_payday_end', 'is_quarter_end', 'days_to_tet', 'event_score', 'cogs_profile']
        cats = [f'{p}_{c.lower()}' for c in self.categories_ for p in ['share', 'event_score', 'inter']]
        cyclic = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
        return base + cats + cyclic + ['prev_q4_momentum', 'cat_blended_momentum', 'peak_momentum', 'is_odd_year_aug', 'is_double_day']

    def get_feature_names_out(self, input_features=None):
        """Standard scikit-learn 1.0+ API for feature names."""
        lag_cols = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7']
        return np.array(lag_cols + self.get_feature_names())
