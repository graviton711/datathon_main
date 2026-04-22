import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
        self.event_score_map = {} # (month, day) -> median_lift
        
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

    def fit(self, X, y=None):
        # We need Revenue (y) to calculate historical signals
        if y is not None:
            df = X.copy()
            df[self.rev_col] = y
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            
            self.start_date_ref = df[self.date_col].min()
            
            df['year'] = df[self.date_col].dt.year
            df['month'] = df[self.date_col].dt.month
            df['day'] = df[self.date_col].dt.day
            
            years = sorted(df['year'].unique())
            
            # Pre-calculate Q4 totals for momentum
            q4_totals = {}
            for yr in years:
                total = df[(df['year'] == yr) & (df['month'] >= 10)][self.rev_col].sum()
                q4_totals[yr] = total
                
            for yr in years:
                if (yr - 1) in q4_totals and (yr - 2) in q4_totals and q4_totals[yr - 2] > 0:
                    self.q4_momentum_dict[yr] = (q4_totals[yr - 1] / q4_totals[yr - 2]) - 1
                else:
                    self.q4_momentum_dict[yr] = 0.0
            
            # 4. Pure Signal Discovery (Rule 10 compliant)
            self._discover_event_scores(df)
        
        return self

    def _get_days_to_tet(self, row_date):
        """Helper to calculate distance to closest Lunar New Year."""
        yr = row_date.year
        candidates = []
        for y_off in [-1, 0, 1]:
            if (yr + y_off) in self.tet_dates:
                candidates.append(self.tet_dates[yr + y_off])
        
        if not candidates: return 999
        
        diffs = [(row_date - t).days for t in candidates]
        closest_diff = min(diffs, key=abs)
        
        if -60 <= closest_diff <= 20:
            return closest_diff
        return 999

    def _discover_event_scores(self, df):
        """
        Pure Signal Discovery: Scans all 366 days and identifies consistent lifts 
        that are NOT caused by Tet.
        """
        print("Starting Pure Signal Discovery (Scanning 366 days)...")
        
        # Pre-calculate days_to_tet for the training set to speed up filtering
        df['days_to_tet'] = df[self.date_col].apply(self._get_days_to_tet)
        
        for m in range(1, 13):
            for d in range(1, 32):
                # Find all occurrences of this (month, day)
                target_mask = (df['month'] == m) & (df['day'] == d)
                if not target_mask.any(): continue
                
                occurrences = df[target_mask]
                
                # IMPORTANT: Filter out years where this day is 'Contaminated' by Tet
                # We only want 'Pure' signals (e.g. Payday, Fixed Holidays)
                pure_occurrences = occurrences[occurrences['days_to_tet'].abs() > 15]
                
                if len(pure_occurrences) < 5: continue # Need at least 5 'pure' years to be sure
                
                lifts = []
                for _, row in pure_occurrences.iterrows():
                    yr = row['year']
                    # Monthly baseline (EXCLUDING a window around the target date)
                    m_mask = (df['year'] == yr) & (df['month'] == m)
                    window_dates = pd.date_range(row[self.date_col] - pd.Timedelta(days=10), 
                                                 row[self.date_col] + pd.Timedelta(days=10))
                    baseline = df[m_mask & (~df[self.date_col].isin(window_dates))][self.rev_col].mean()
                    
                    if baseline > 0:
                        lifts.append(row[self.rev_col] / baseline)
                
                if lifts:
                    median_l = np.median(lifts)
                    if median_l > 1.10: # Increased sensitivity to Legacy Events
                        self.event_score_map[(m, d)] = median_l

        if self.event_score_map:
            print(f"Discovery complete: Found {len(self.event_score_map)} pure event signals.")

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
        
        X['days_to_tet'] = X[self.date_col].apply(self._get_days_to_tet)
        
        # Event Score Feature
        X['event_score'] = X[self.date_col].apply(lambda d: self.event_score_map.get((d.month, d.day), 1.0))
        
        # 3. Dynamic Trailing Momentum (prev_q4_momentum)
        X['prev_q4_momentum'] = X[self.date_col].dt.year.map(self.q4_momentum_dict).fillna(0.0)
            
        # Drop non-feature columns
        cols_to_drop = [self.date_col]
        if self.rev_col in X.columns:
            cols_to_drop.append(self.rev_col)
            
        X = X.drop(columns=cols_to_drop)
        return X

    def get_feature_names(self):
        return ['month', 'day', 'day_of_week', 'is_weekend', 'days_to_tet', 'event_score', 'prev_q4_momentum']
