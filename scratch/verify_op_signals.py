import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_op_signals():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    inventory = pd.read_parquet(DATA_DIR / "inventory.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    
    # Pre-process
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    inventory['Date'] = pd.to_datetime(inventory['snapshot_date'])
    inventory['year'] = inventory['Date'].dt.year
    inventory['month'] = inventory['Date'].dt.month
    
    returns['Date'] = pd.to_datetime(returns['return_date'])
    returns['year'] = returns['Date'].dt.year
    returns['month'] = returns['Date'].dt.month

    # 2. Evaluation Folds
    folds = [2020, 2021, 2022]
    results = []
    
    for test_yr in folds:
        train_yr_max = test_yr - 1
        train_raw = sales[sales['year'] <= train_yr_max].copy()
        test_raw = sales[sales['year'] == test_yr].copy()
        
        # --- SCALE PROJECTION (No Leak) ---
        train_medians = train_raw.groupby('year')['Revenue'].median().to_dict()
        q4_last = train_raw[(train_raw['year'] == train_yr_max) & (train_raw['month'] >= 10)]['Revenue'].sum()
        q4_prev = train_raw[(train_raw['year'] == train_yr_max - 1) & (train_raw['month'] >= 10)]['Revenue'].sum()
        projected_median = train_medians[train_yr_max] * (q4_last / (q4_prev + 1e-6))
        
        # --- OP SIGNALS (Q4 of previous year) ---
        # Stockout Rate
        q4_inv = inventory[(inventory['year'] <= train_yr_max) & (inventory['month'] >= 10)]
        stockout_map = q4_inv.groupby('year')['stockout_flag'].mean().to_dict()
        
        # Return Rate
        q4_ret = returns[(returns['year'] <= train_yr_max) & (returns['month'] >= 10)]
        ret_sum = q4_ret.groupby('year')['return_quantity'].sum()
        sold_sum = q4_inv.groupby('year')['units_sold'].sum()
        return_map = (ret_sum / (sold_sum + 1e-6)).to_dict()
        
        # --- FEATURE PREPARATION ---
        def prep_features(df_raw, is_test=False):
            df = df_raw.copy()
            df['y'] = df['Revenue'] / (projected_median if is_test else df['year'].map(train_medians))
            df['month_feat'] = df['month']
            df['rev_lag_1_norm'] = (df['Revenue'].shift(1).fillna(method='bfill') / (projected_median if is_test else df['year'].map(train_medians)))
            df['stockout_signal'] = df['year'].map(lambda y: stockout_map.get(y-1, 0.0))
            df['return_signal'] = df['year'].map(lambda y: return_map.get(y-1, 0.0))
            return df.dropna()

        train_df = prep_features(train_raw)
        test_df = prep_features(test_raw, is_test=True)
        
        cols_base = ['month_feat', 'rev_lag_1_norm']
        cols_enh = cols_base + ['stockout_signal', 'return_signal']
        
        m_base = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        m_base.fit(train_df[cols_base], train_df['y'])
        p_base = m_base.predict(test_df[cols_base]) * projected_median
        mae_base = mean_absolute_error(test_raw['Revenue'], p_base)
        
        m_enh = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        m_enh.fit(train_df[cols_enh], train_df['y'])
        p_enh = m_enh.predict(test_df[cols_enh]) * projected_median
        mae_enh = mean_absolute_error(test_raw['Revenue'], p_enh)
        
        results.append({'fold': test_yr, 'base': mae_base, 'enh': mae_enh})
        
    res_df = pd.DataFrame(results)
    avg_base = res_df['base'].mean()
    avg_enh = res_df['enh'].mean()
    
    print("--- Operational Signal Verification (3-Fold, No Leak) ---")
    print(f"Base MAE: {avg_base:,.0f}")
    print(f"Enh MAE (Inventory/Returns): {avg_enh:,.0f}")
    print(f"Gain: {avg_base - avg_enh:,.0f} ({(1 - avg_enh/avg_base)*100:.2f}%)")

if __name__ == "__main__":
    verify_op_signals()
