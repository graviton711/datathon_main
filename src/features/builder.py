import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import Config

class BaselineFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A standard Scikit-Learn compatible transformer for baseline feature extraction.
    This creates the absolute minimum features needed for a time-series baseline.
    """
    def __init__(self, date_col='Date', rev_col='Revenue'):
        self.date_col = date_col
        self.rev_col = rev_col
        self.start_date_ref = None
        self.q4_momentum_dict = {}
        self.q4_momentum_default = 0.0
        self.event_score_map = {} # (month, day) -> median_lift
        self.category_event_map = {} # category -> {(month, day) -> median_lift}
        self.category_profile_map = {} # month -> {cat: share}
        self.cogs_monthly_profile = {} # month -> median_ratio
        self.categories_ = [] # Dynamic list of categories
        self.latest_peak_lift = 1.0 # Momentum of the last major campaign
        
        # Lunar New Year (Mung 1) dictionary
        self.tet_dates = {
            2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31',
            2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
            2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
            2021: '2021-02-12', 2022: '2022-02-01', 2023: '2023-01-22',
            2024: '2024-02-10'
        }
        # Convert to datetime objects for faster lookup
        self.tet_dates = {k: pd.to_datetime(v) for k, v in self.tet_dates.items()}

    def set_q4_momentum_map(self, q4_momentum_dict, default_value=None):
        """Inject externally computed Q4 momentum based on raw revenue."""
        if q4_momentum_dict is None:
            self.q4_momentum_dict = {}
        else:
            self.q4_momentum_dict = {int(k): float(v) for k, v in q4_momentum_dict.items()}

        if default_value is None:
            vals = list(self.q4_momentum_dict.values())
            self.q4_momentum_default = float(np.median(vals)) if vals else 0.0
        else:
            self.q4_momentum_default = float(default_value)

        return self

    def set_cogs_monthly_profile(self, profile_map):
        """Inject externally computed COGS monthly ratio profiles."""
        self.cogs_monthly_profile = {int(k): float(v) for k, v in profile_map.items()}
        return self

    def set_category_event_map(self, category_event_map):
        """Inject category-specific event lift maps."""
        self.category_event_map = category_event_map
        return self

    def fit(self, X, y=None):
        # We need Revenue (y) to calculate historical signals
        if y is not None:
            df = X.copy()
            df[self.rev_col] = y
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            
            self.start_date_ref = df[self.date_col].min()
            self.max_date_ref = df[self.date_col].max()
            
            df['year'] = df[self.date_col].dt.year
            df['month'] = df[self.date_col].dt.month
            df['day'] = df[self.date_col].dt.day
            
            # 4. Pure Signal Discovery (Rule 10 compliant)
            if not self.event_score_map:
                self._discover_event_scores(df)
            # 5. Category Profile Discovery
            if not self.category_profile_map:
                self._discover_category_profiles(self.max_date_ref)
            # 6. Peak Momentum Discovery
            self._discover_peak_momentum(df)
        
        return self

    def _vectorize_days_to_tet(self, dates):
        """Vectorized calculation of distance to closest Lunar New Year."""
        tet_dates = np.array(list(self.tet_dates.values()), dtype='datetime64[ns]')
        # Broadcast subtraction: (N, 1) - (1, T) -> (N, T)
        dates_np = dates.values.astype('datetime64[ns]')[:, np.newaxis]
        diffs = (dates_np - tet_dates).astype('timedelta64[D]').astype(int)
        
        # Find index of the closest Tet date for each row
        min_idx = np.argmin(np.abs(diffs), axis=1)
        closest_diff = diffs[np.arange(len(dates)), min_idx]
        
        # Apply threshold and return
        return np.where((closest_diff >= -60) & (closest_diff <= 20), closest_diff, 999)

    def _resolve_prev_q4_momentum(self, year):
        if year in self.q4_momentum_dict:
            return self.q4_momentum_dict[year]

        # Carry forward the latest known momentum for future years.
        known_years = [y for y in self.q4_momentum_dict if y < year]
        if known_years:
            return self.q4_momentum_dict[max(known_years)]

        return self.q4_momentum_default

    def _discover_peak_momentum(self, df):
        """
        Identifies the strength of the last major campaign to carry forward as momentum.
        """
        # Find days with significant lift (> 2.0)
        # We use the lift already calculated in _discover_event_scores if possible, 
        # or recalculate it here relative to yearly median for robustness.
        df = df.copy().sort_values(self.date_col)
        yearly_medians = df.groupby(df[self.date_col].dt.year)[self.rev_col].transform('median')
        df['rel_lift'] = df[self.rev_col] / (yearly_medians + 1e-6)
        
        peaks = df[df['rel_lift'] > 2.0]
        if not peaks.empty:
            self.latest_peak_lift = float(peaks['rel_lift'].iloc[-1])
            # Clip to reasonable range to avoid extreme outliers
            self.latest_peak_lift = np.clip(self.latest_peak_lift, 1.0, 10.0)
            print(f"Peak Momentum Discovered: {self.latest_peak_lift:.2f}x (from {peaks[self.date_col].iloc[-1].date()})")
        else:
            self.latest_peak_lift = 1.0

    def _discover_event_scores(self, df):
        """
        Pure Signal Discovery: Vectorized scan of all days to identify consistent lifts.
        """
        print("Starting Optimized Signal Discovery...")
        
        # 1. Pre-calculate necessary components
        df = df.copy()
        df['days_to_tet'] = self._vectorize_days_to_tet(df[self.date_col])
        
        # 2. Calculate Monthly Baseline (Mean of each month)
        monthly_baseline = df.groupby(['year', 'month'])[self.rev_col].transform('mean')
        df['lift'] = df[self.rev_col] / (monthly_baseline + 1e-6)
        
        # 3. Filter for 'Pure' days (not contaminated by Tet)
        pure_df = df[df['days_to_tet'].abs() > Config.TET_CONTAMINATION_DAYS].copy()
        
        # 4. Group by (month, day) to find consistent lifts
        stats = pure_df.groupby(['month', 'day'])['lift'].agg(['median', 'count'])
        
        # 5. Filter for signals with enough occurrences and significant lift
        # Rule: At least Config.EVENT_MIN_OCCURRENCES 'pure' years and median lift > Config.EVENT_LIFT_THRESHOLD
        signals = stats[(stats['count'] >= Config.EVENT_MIN_OCCURRENCES) & (stats['median'] > Config.EVENT_LIFT_THRESHOLD)]
        
        self.event_score_map = signals['median'].to_dict()

        if self.event_score_map:
            print(f"Discovery complete: Found {len(self.event_score_map)} pure event signals.")

    def _discover_category_profiles(self, max_date=None):
        """
        Calculates historical monthly revenue shares for each category.
        This captures the 'Category Mix' signal which is highly correlated with total revenue.
        """
        print("Starting Category Profile Discovery...")
        try:
            # 1. Load raw data for mix calculation
            orders = pd.read_parquet(Config.DATA_DIR / "processed" / "orders.parquet")[['order_id', 'order_date']]
            
            orders['order_date'] = pd.to_datetime(orders['order_date'])
            if max_date is not None:
                orders = orders[orders['order_date'] <= max_date]
                
            items = pd.read_parquet(Config.DATA_DIR / "processed" / "order_items.parquet")[['order_id', 'product_id', 'quantity', 'unit_price', 'discount_amount']]
            products = pd.read_parquet(Config.DATA_DIR / "processed" / "products.parquet")[['product_id', 'category']]
            
            # 2. Merge and calculate revenue
            items = pd.merge(items, orders, on='order_id')
            items = pd.merge(items, products, on='product_id')
            items['item_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
            items['month'] = items['order_date'].dt.month
            
            # 3. Calculate Monthly Shares
            self.categories_ = sorted(items['category'].unique().tolist())
            cat_monthly = items.groupby(['month', 'category'])['item_rev'].sum().unstack().fillna(0)
            cat_shares = cat_monthly.div(cat_monthly.sum(axis=1), axis=0)
            
            self.category_profile_map = cat_shares.to_dict(orient='index')
            print(f"Category discovery complete: Profiles for {len(self.category_profile_map)} months created.")
            
        except Exception as e:
            print(f"Warning: Category discovery failed ({e}). Using empty profiles.")
            self.category_profile_map = {}

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
        X['is_wednesday'] = (X['day_of_week'] == 2).astype(int)
        X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
        
        # 2. Payday & Quarter Signals (Refined based on verified intra-month seasonality)
        # Day 1 is the peak residual payday. Day 4 is the structural slump (stockout/fatigue).
        X['is_payday_peak'] = ((X['day'] >= 1) & (X['day'] <= 2)).astype(int)
        X['is_payday_slump'] = ((X['day'] >= 3) & (X['day'] <= 6)).astype(int)
        X['is_payday_end'] = (X['day'] >= 27).astype(int)
        
        # Quarter end: Last 7 days of months 3, 6, 9, 12
        is_q_month = X['month'].isin([3, 6, 9, 12])
        is_last_week = X['day'] >= 24
        X['is_quarter_end'] = (is_q_month & is_last_week).astype(int)
        
        X['days_to_tet'] = self._vectorize_days_to_tet(X[self.date_col])
        
        # Event Score Feature: Use vectorized mapping
        # Create a day-of-year key (MMDD) for faster lookup
        X['mmdd'] = X['month'] * 100 + X['day']
        event_map_vec = {m * 100 + d: score for (m, d), score in self.event_score_map.items()}
        X['event_score'] = X['mmdd'].map(event_map_vec).fillna(1.0)
        
        # 2.5 Category Mix & Specific Event Signals
        # Pre-calculate category share mapping dynamically
        for cat in self.categories_:
            cat_lower = cat.lower()
            cat_share_col = f'share_{cat_lower}'
            
            # Month share
            cat_map = {month: shares.get(cat, 0.0) for month, shares in self.category_profile_map.items()}
            X[cat_share_col] = X['month'].map(cat_map).fillna(0.0)
            
            # Category-specific Event Score
            cat_event_map = self.category_event_map.get(cat, {})
            cat_event_map_vec = {m * 100 + d: score for (m, d), score in cat_event_map.items()}
            cat_event_col = f'event_score_{cat_lower}'
            X[cat_event_col] = X['mmdd'].map(cat_event_map_vec).fillna(1.0)
            
            # Interaction: How much this event impacts THIS specific category today
            X[f'inter_{cat_lower}'] = X['event_score'] * X[cat_share_col]
        
        # COGS Ratio Profile
        X['cogs_profile'] = X['month'].map(self.cogs_monthly_profile).fillna(0.85)
        
        # 3. Dynamic Trailing Momentum (prev_q4_momentum)
        # Pre-calculate a full map for all years in X to avoid expensive fallback logic per row
        unique_years = X['year'].unique()
        year_momentum_map = {y: self._resolve_prev_q4_momentum(y) for y in unique_years}
        X['prev_q4_momentum'] = X['year'].map(year_momentum_map)

        # 4. Peak Momentum Signal (Dynamic)
        # We need a running last peak lift that updates within the dataset
        if self.rev_col in X.columns:
            # During training/fit
            X_tmp = X.copy()
            # Calculate a quick monthly-baseline lift for momentum discovery
            monthly_m = X_tmp.groupby(['year', 'month'])[self.rev_col].transform('mean')
            X_tmp['lift_tmp'] = X_tmp[self.rev_col] / (monthly_m + 1e-6)
            
            X_tmp['is_peak_tmp'] = (X_tmp['lift_tmp'] > 2.0)
            X_tmp['peak_val_tmp'] = np.where(X_tmp['is_peak_tmp'], X_tmp['lift_tmp'], np.nan)
            X['peak_momentum_val'] = pd.Series(X_tmp['peak_val_tmp']).shift(1).ffill().fillna(1.0).values
        else:
            # During predict/inference
            X['peak_momentum_val'] = self.latest_peak_lift

        X['peak_momentum'] = X['event_score'] * X['peak_momentum_val']
        X = X.drop(columns=['peak_momentum_val'])

        # 5. Precise Biennial Signal (Urban Blowout Gate)
        # Only active in August of odd years.
        X['is_odd_year_aug'] = ((X['year'] % 2 != 0) & (X['month'] == 8)).astype(int)

        # Add Cyclic Encoding for Day and Month
        days_in_month = X[self.date_col].dt.days_in_month
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / days_in_month)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / days_in_month)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
            
        # Drop non-feature columns
        cols_to_drop = [self.date_col, 'mmdd']
        cols_to_drop.append('year')
        if self.rev_col in X.columns:
            cols_to_drop.append(self.rev_col)
            
        X = X.drop(columns=cols_to_drop)

        ordered_cols = ['rev_lag_1', 'rev_lag_7', 'rev_roll_7'] + self.get_feature_names()
        if all(col in X.columns for col in ordered_cols):
            X = X[ordered_cols]

        return X

    def get_feature_names(self):
        base_features = ['month', 'day', 'day_of_week', 'is_wednesday', 'is_weekend', 
                         'is_payday_peak', 'is_payday_slump', 'is_payday_end', 'is_quarter_end',
                         'days_to_tet', 'event_score', 'cogs_profile']
        
        cat_features = []
        for cat in self.categories_:
            cat_lower = cat.lower()
            cat_features.extend([f'share_{cat_lower}', f'event_score_{cat_lower}', f'inter_{cat_lower}'])
        
        # Cyclic features are added after category features
        cyclic_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        # prev_q4_momentum and peak_momentum are added last
        return base_features + cat_features + cyclic_features + ['prev_q4_momentum', 'peak_momentum', 'is_odd_year_aug']
