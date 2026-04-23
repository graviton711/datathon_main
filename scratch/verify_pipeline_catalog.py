import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_pipeline_with_catalog():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date']]
    orders['Date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['Date'].dt.year
    orders['month'] = orders['Date'].dt.month
    
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")[['order_id', 'product_id']]
    df_items = pd.merge(items, orders, on='order_id')
    
    # 2. Define Evaluate Function
    def run_fold(test_yr, use_catalog=False):
        train_yr_max = test_yr - 1
        train_raw = sales[sales['year'] <= train_yr_max].copy()
        test_raw = sales[sales['year'] == test_yr].copy()
        
        # --- INERTIA CALIBRATION ---
        annual_medians = train_raw.groupby('year')['Revenue'].median()
        q4_rev = train_raw[train_raw['month'] >= 10].groupby('year')['Revenue'].sum()
        q4_orders = orders[(orders['year'] <= train_yr_max) & (orders['month'] >= 10)].groupby('year').size()
        q4_catalog = df_items[(df_items['year'] <= train_yr_max) & (df_items['month'] >= 10)].groupby('year')['product_id'].nunique()
        
        # Regression to find weights
        avail = sorted(list(set(q4_rev.index) & set(q4_orders.index) & set(q4_catalog.index) & set(annual_medians.index)))
        rows = []
        for i in range(2, len(avail)):
            y_val = np.log(annual_medians[avail[i]] / annual_medians[avail[i-1]])
            m_rev = np.log(q4_rev[avail[i-1]] / q4_rev[avail[i-2]])
            m_ord = np.log(q4_orders[avail[i-1]] / q4_orders[avail[i-2]])
            m_aov = m_rev - m_ord
            m_cat = np.log(q4_catalog[avail[i-1]] / q4_catalog[avail[i-2]])
            rows.append({'y': y_val, 'x_rev': m_rev, 'x_ord': m_ord, 'x_aov': m_aov, 'x_cat': m_cat})
        
        reg_df = pd.DataFrame(rows)
        import statsmodels.api as sm
        cols = ['x_rev', 'x_ord', 'x_aov', 'x_cat'] if use_catalog else ['x_rev', 'x_ord', 'x_aov']
        X = sm.add_constant(reg_df[cols])
        model_inertia = sm.OLS(reg_df['y'], X).fit()
        
        # Project Scale for test_yr
        m_rev_test = np.log(q4_rev[train_yr_max] / q4_rev[train_yr_max-1])
        m_ord_test = np.log(q4_orders[train_yr_max] / q4_orders[train_yr_max-1])
        m_aov_test = m_rev_test - m_ord_test
        m_cat_test = np.log(q4_catalog[train_yr_max] / q4_catalog[train_yr_max-1])
        
        test_input = [1, m_rev_test, m_ord_test, m_aov_test, m_cat_test] if use_catalog else [1, m_rev_test, m_ord_test, m_aov_test]
        projected_log_g = np.dot(model_inertia.params, test_input)
        projected_median = annual_medians[train_yr_max] * np.exp(projected_log_g)
        
        # --- TRAIN SHAPE MODEL (LGBM) ---
        train_raw['y_norm'] = train_raw['Revenue'] / train_raw['year'].map(annual_medians)
        train_raw['rev_lag_1'] = train_raw['y_norm'].shift(1).fillna(method='bfill')
        
        lgbm = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        lgbm.fit(train_raw[['month', 'rev_lag_1']], train_raw['y_norm'])
        
        # --- PREDICT ---
        test_raw['rev_lag_1'] = (test_raw['Revenue'].shift(1).fillna(train_raw['Revenue'].iloc[-1]) / projected_median)
        preds = lgbm.predict(test_raw[['month', 'rev_lag_1']]) * projected_median
        
        return mean_absolute_error(test_raw['Revenue'], preds)

    # 3. Run Comparison on Fold 2022
    mae_base = run_fold(2022, use_catalog=False)
    mae_enh = run_fold(2022, use_catalog=True)
    
    print("--- Full Pipeline Catalog Verification (Fold 2022) ---")
    print(f"Baseline MAE: {mae_base:,.0f}")
    print(f"Enhanced MAE (with Catalog Inertia): {mae_enh:,.0f}")
    print(f"Gain: {mae_base - mae_enh:,.0f} ({(1 - mae_enh/mae_base)*100:.2f}%)")

if __name__ == "__main__":
    verify_pipeline_with_catalog()
