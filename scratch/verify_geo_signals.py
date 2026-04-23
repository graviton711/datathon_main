import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_geo_signals():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")
    geo = pd.read_parquet(DATA_DIR / "geography.parquet")
    
    # Pre-process
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    orders['Date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['Date'].dt.year
    orders['month'] = orders['Date'].dt.month
    
    # Link Orders to Geography
    order_geo = pd.merge(orders[['order_id', 'year', 'month', 'zip']], geo[['zip', 'region']], on='zip', how='left')
    
    # 2. Evaluation Folds
    folds = [2020, 2021, 2022]
    total_mae_base = 0
    total_mae_enh = 0
    
    for test_yr in folds:
        train_yr_max = test_yr - 1
        train_raw = sales[sales['year'] <= train_yr_max].copy()
        test_raw = sales[sales['year'] == test_yr].copy()
        
        # --- SCALE PROJECTION (No Leak) ---
        train_medians = train_raw.groupby('year')['Revenue'].median().to_dict()
        q4_last = train_raw[(train_raw['year'] == train_yr_max) & (train_raw['month'] >= 10)]['Revenue'].sum()
        q4_prev = train_raw[(train_raw['year'] == train_yr_max - 1) & (train_raw['month'] >= 10)]['Revenue'].sum()
        projected_median = train_medians[train_yr_max] * (q4_last / q4_prev)
        
        # --- GEO SIGNALS (Q4 Share of previous year) ---
        # Calculate Q4 shares for all years up to training max
        q4_orders = order_geo[(order_geo['year'] <= train_yr_max) & (order_geo['month'] >= 10)]
        yearly_q4_geo = q4_orders.groupby(['year', 'region']).size().unstack(fill_value=0)
        yearly_q4_share = yearly_q4_geo.div(yearly_q4_geo.sum(axis=1), axis=0)
        
        # Map signal to years (For 2022, use Q4 2021 share)
        central_share_map = yearly_q4_share['Central'].to_dict() if 'Central' in yearly_q4_share.columns else {}
        west_share_map = yearly_q4_share['West'].to_dict() if 'West' in yearly_q4_share.columns else {}
        
        # --- FEATURE PREPARATION ---
        def prep_features(df_raw, is_test=False):
            df = df_raw.copy()
            df['y'] = df['Revenue'] / (projected_median if is_test else df['year'].map(train_medians))
            df['month_feat'] = df['month']
            df['rev_lag_1_norm'] = (df['Revenue'].shift(1).fillna(method='bfill') / (projected_median if is_test else df['year'].map(train_medians)))
            # The signal for year Y is the Q4 share of year Y-1
            df['central_share'] = df['year'].map(lambda y: central_share_map.get(y-1, 0.3))
            df['west_share'] = df['year'].map(lambda y: west_share_map.get(y-1, 0.25))
            return df.dropna()

        train_df = prep_features(train_raw)
        test_df = prep_features(test_raw, is_test=True)
        
        base_cols = ['month_feat', 'rev_lag_1_norm']
        enh_cols = base_cols + ['central_share', 'west_share']
        
        # Train & Predict
        m_base = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        m_base.fit(train_df[base_cols], train_df['y'])
        p_base = m_base.predict(test_df[base_cols]) * projected_median
        
        m_enh = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        m_enh.fit(train_df[enh_cols], train_df['y'])
        p_enh = m_enh.predict(test_df[enh_cols]) * projected_median
        
        total_mae_base += mean_absolute_error(test_raw['Revenue'], p_base)
        total_mae_enh += mean_absolute_error(test_raw['Revenue'], p_enh)
        
    avg_base = total_mae_base / len(folds)
    avg_enh = total_mae_enh / len(folds)
    
    print("--- Geography Signal Verification (3-Fold, No Leak) ---")
    print(f"Base MAE: {avg_base:,.0f}")
    print(f"Enh MAE (with Geo Shares): {avg_enh:,.0f}")
    print(f"Gain: {avg_base - avg_enh:,.0f} ({(1 - avg_enh/avg_base)*100:.2f}%)")

if __name__ == "__main__":
    verify_geo_signals()
