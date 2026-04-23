import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_category_recency():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    products = pd.read_parquet(DATA_DIR / "products.parquet")[['product_id', 'category']]
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date']]
    
    items['rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
    df_cat = pd.merge(items, products, on='product_id')
    df_cat = pd.merge(df_cat, orders, on='order_id')
    df_cat['Date'] = pd.to_datetime(df_cat['order_date'])
    df_cat['year'] = df_cat['Date'].dt.year
    df_cat['month'] = df_cat['Date'].dt.month

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
        
        # --- CATEGORY SIGNALS ---
        df_train_cat = df_cat[df_cat['year'] <= train_yr_max]
        
        # SCENARIO 1: Global Mix
        cat_global = df_train_cat.groupby(['month', 'category'])['rev'].sum().unstack(fill_value=0)
        share_global = cat_global.div(cat_global.sum(axis=1), axis=0).to_dict('index')
        
        # SCENARIO 2: Recent Mix (Last 2 years only)
        df_recent = df_train_cat[df_train_cat['year'] >= train_yr_max - 1]
        cat_recent = df_recent.groupby(['month', 'category'])['rev'].sum().unstack(fill_value=0)
        share_recent = cat_recent.div(cat_recent.sum(axis=1), axis=0).to_dict('index')
        
        # --- EVALUATE ---
        def run_eval(share_map):
            train_feat = train_raw.copy()
            train_feat['y'] = train_feat['Revenue'] / train_feat['year'].map(train_medians)
            for cat in ['Casual', 'Streetwear', 'Outdoor', 'GenZ']:
                train_feat[f'share_{cat.lower()}'] = train_feat['month'].map(lambda m: share_map.get(m, {}).get(cat, 0.0))
            
            train_df = train_feat.dropna()
            cols = [f'share_{cat.lower()}' for cat in ['Casual', 'Streetwear', 'Outdoor', 'GenZ']]
            
            model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            model.fit(train_df[cols], train_df['y'])
            
            test_feat = test_raw.copy()
            for cat in ['Casual', 'Streetwear', 'Outdoor', 'GenZ']:
                test_feat[f'share_{cat.lower()}'] = test_feat['month'].map(lambda m: share_map.get(m, {}).get(cat, 0.0))
            
            preds = model.predict(test_feat[cols]) * projected_median
            return mean_absolute_error(test_raw['Revenue'], preds)

        mae_global = run_eval(share_global)
        mae_recent = run_eval(share_recent)
        
        results.append({'fold': test_yr, 'global': mae_global, 'recent': mae_recent})
        
    res_df = pd.DataFrame(results)
    avg_global = res_df['global'].mean()
    avg_recent = res_df['recent'].mean()
    
    print("--- Category Mix Verification (3-Fold, No Leak) ---")
    print(f"Base MAE (Global Mix): {avg_global:,.0f}")
    print(f"Recent MAE (Last 2 Years): {avg_recent:,.0f}")
    print(f"Gain: {avg_global - avg_recent:,.0f} ({(1 - avg_recent/avg_global)*100:.2f}%)")

if __name__ == "__main__":
    verify_category_recency()
