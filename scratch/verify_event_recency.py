import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_event_recency():
    # 1. Load Data
    df = pd.read_parquet(DATA_DIR / "sales.parquet")
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['mmdd'] = df['Date'].dt.strftime('%m-%d')
    
    # 2. Evaluation Folds
    folds = [2020, 2021, 2022]
    results = []
    
    for test_yr in folds:
        train_yr_max = test_yr - 1
        train_raw = df[df['year'] <= train_yr_max].copy()
        test_raw = df[df['year'] == test_yr].copy()
        
        # --- SCALE PROJECTION (No Leak) ---
        train_medians = train_raw.groupby('year')['Revenue'].median().to_dict()
        q4_last = train_raw[(train_raw['year'] == train_yr_max) & (train_raw['month'] >= 10)]['Revenue'].sum()
        q4_prev = train_raw[(train_raw['year'] == train_yr_max - 1) & (train_raw['month'] >= 10)]['Revenue'].sum()
        projected_median = train_medians[train_yr_max] * (q4_last / (q4_prev + 1e-6))
        
        # Calculate Lifts for discovery
        train_raw['rev_norm'] = train_raw['Revenue'] / train_raw['year'].map(train_medians)
        monthly_baseline = train_raw.groupby(['year', 'month'])['rev_norm'].transform('mean')
        train_raw['lift'] = train_raw['rev_norm'] / (monthly_baseline + 1e-6)
        
        # --- SCENARIO 1: Global Median ---
        score_map_global = train_raw.groupby('mmdd')['lift'].median().to_dict()
        
        # --- SCENARIO 2: Recency-Weighted Score ---
        # 70% weight to last 2 years, 30% to the rest
        recent_years = [train_yr_max, train_yr_max - 1]
        score_recent = train_raw[train_raw['year'].isin(recent_years)].groupby('mmdd')['lift'].median()
        score_old = train_raw[~train_raw['year'].isin(recent_years)].groupby('mmdd')['lift'].median()
        
        score_map_weighted = {}
        for mmdd in train_raw['mmdd'].unique():
            s_recent = score_recent.get(mmdd, 1.0)
            s_old = score_old.get(mmdd, 1.0)
            score_map_weighted[mmdd] = 0.7 * s_recent + 0.3 * s_old
            
        # --- EVALUATE ---
        def run_eval(score_map):
            train_feat = train_raw.copy()
            train_feat['event_score'] = train_feat['mmdd'].map(score_map).fillna(1.0)
            train_feat['rev_lag_1_norm'] = (train_feat['Revenue'].shift(1).fillna(method='bfill') / train_feat['year'].map(train_medians))
            train_df = train_feat.dropna()
            
            cols = ['event_score', 'rev_lag_1_norm']
            model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            model.fit(train_df[cols], train_df['rev_norm'])
            
            test_feat = test_raw.copy()
            test_feat['event_score'] = test_feat['mmdd'].map(score_map).fillna(1.0)
            test_feat['rev_lag_1_norm'] = (test_feat['Revenue'].shift(1).fillna(train_raw['Revenue'].iloc[-1]) / projected_median)
            
            preds = model.predict(test_feat[cols]) * projected_median
            return mean_absolute_error(test_raw['Revenue'], preds)

        mae_global = run_eval(score_map_global)
        mae_weighted = run_eval(score_map_weighted)
        
        results.append({'fold': test_yr, 'global': mae_global, 'weighted': mae_weighted})
        
    res_df = pd.DataFrame(results)
    avg_global = res_df['global'].mean()
    avg_weighted = res_df['weighted'].mean()
    
    print("--- Event Recency Verification (3-Fold, No Leak) ---")
    print(f"Base MAE (Global Median): {avg_global:,.0f}")
    print(f"Weighted MAE (Recency-Aware): {avg_weighted:,.0f}")
    print(f"Gain: {avg_global - avg_weighted:,.0f} ({(1 - avg_weighted/avg_global)*100:.2f}%)")

if __name__ == "__main__":
    verify_event_recency()
