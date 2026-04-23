import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def signal_hunt_no_leak():
    # 1. Load All Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    traffic = pd.read_parquet(DATA_DIR / "web_traffic.parquet")
    reviews = pd.read_parquet(DATA_DIR / "reviews.parquet")
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    
    # Pre-process dates
    for d in [sales, traffic, reviews, orders]:
        date_col = 'Date' if 'Date' in d.columns else ('date' if 'date' in d.columns else 'review_date' if 'review_date' in d.columns else 'order_date')
        d['Date'] = pd.to_datetime(d[date_col])
        d['year'] = d['Date'].dt.year
        d['month'] = d['Date'].dt.month

    # 2. Define Evaluation Loop (2020, 2021, 2022)
    folds = [2020, 2021, 2022]
    
    def evaluate_signals(signal_name, get_signal_fn):
        total_mae_base = 0
        total_mae_enh = 0
        
        for test_yr in folds:
            train_yr_max = test_yr - 1
            train_raw = sales[sales['year'] <= train_yr_max].copy()
            test_raw = sales[sales['year'] == test_yr].copy()
            
            # --- LEAK-FREE SCALE PROJECTION ---
            train_medians = train_raw.groupby('year')['Revenue'].median().to_dict()
            last_yr = train_yr_max
            prev_yr = train_yr_max - 1
            
            q4_last = train_raw[(train_raw['year'] == last_yr) & (train_raw['month'] >= 10)]['Revenue'].sum()
            q4_prev = train_raw[(train_raw['year'] == prev_yr) & (train_raw['month'] >= 10)]['Revenue'].sum()
            growth = q4_last / (q4_prev + 1e-6)
            projected_median = train_medians[last_yr] * growth
            
            # --- FEATURE PREPARATION ---
            train_feat = train_raw.copy()
            train_feat['y'] = train_feat['Revenue'] / train_feat['year'].map(train_medians)
            train_feat['month_feat'] = train_feat['month']
            train_feat['rev_lag_1_norm'] = (train_feat['Revenue'].shift(1) / train_feat['year'].map(train_medians)).fillna(1.0)
            
            # Add Signal Map
            signal_map = get_signal_fn(train_yr_max)
            train_feat['signal'] = train_feat['year'].map(signal_map).fillna(0.0)
            
            train_df = train_feat.dropna()
            
            # Train
            base_cols = ['month_feat', 'rev_lag_1_norm']
            enh_cols = base_cols + ['signal']
            
            m_base = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            m_base.fit(train_df[base_cols], train_df['y'])
            
            m_enh = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            m_enh.fit(train_df[enh_cols], train_df['y'])
            
            # Test
            test_feat = test_raw.copy()
            test_feat['month_feat'] = test_feat['month']
            test_feat['rev_lag_1_norm'] = (test_feat['Revenue'].shift(1).fillna(train_raw['Revenue'].iloc[-1]) / projected_median)
            test_feat['signal'] = test_feat['year'].map(signal_map).fillna(0.0)
            
            p_base = m_base.predict(test_feat[base_cols]) * projected_median
            p_enh = m_enh.predict(test_feat[enh_cols]) * projected_median
            
            total_mae_base += mean_absolute_error(test_raw['Revenue'], p_base)
            total_mae_enh += mean_absolute_error(test_raw['Revenue'], p_enh)
            
        avg_base = total_mae_base / len(folds)
        avg_enh = total_mae_enh / len(folds)
        return avg_base, avg_enh

    # 3. Define Signals
    def get_traffic_signal(max_yr):
        s = traffic.groupby(['year', 'month'])['sessions'].sum().reset_index()
        q4 = s[s['month'] >= 10].groupby('year')['sessions'].sum()
        res = {}
        for yr in range(2013, max_yr + 2):
            if (yr-1) in q4 and (yr-2) in q4:
                res[yr] = (q4[yr-1] / q4[yr-2]) - 1.0
        return res

    def get_rating_signal(max_yr):
        r = reviews[reviews['month'] >= 10].groupby('year')['rating'].mean()
        res = {}
        for yr in range(2013, max_yr + 2):
            if (yr-1) in r and (yr-2) in r:
                res[yr] = r[yr-1] - r[yr-2]
        return res

    def get_price_signal(max_yr):
        price_data = pd.merge(items, orders[['order_id', 'order_date', 'year', 'month']], on='order_id')
        price_data['rev'] = price_data['quantity'] * price_data['unit_price'] - price_data['discount_amount']
        p = price_data[price_data['month'] >= 10].groupby('year').apply(lambda x: x['rev'].sum() / x['quantity'].sum(), include_groups=False)
        res = {}
        for yr in range(2013, max_yr + 2):
            if (yr-1) in p and (yr-2) in p:
                res[yr] = (p[yr-1] / p[yr-2]) - 1.0
        return res

    print("--- Signal Hunt: Multi-Fold Leak-Free Evaluation ---")
    
    for name, fn in [("Traffic Q4 Momentum", get_traffic_signal), 
                     ("Rating Q4 Shift", get_rating_signal), 
                     ("Price Q4 Momentum", get_price_signal)]:
        base, enh = evaluate_signals(name, fn)
        diff = base - enh
        pct = (diff / base) * 100
        status = "PASS" if diff > 0 else "FAIL"
        print(f"{name}: Base {base:,.0f} -> Enh {enh:,.0f} | Gain: {diff:,.0f} ({pct:.2f}%) | {status}")

if __name__ == "__main__":
    signal_hunt_no_leak()
