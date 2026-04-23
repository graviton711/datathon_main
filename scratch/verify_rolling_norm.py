import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_rolling_norm():
    # 1. Load Data
    df = pd.read_parquet(DATA_DIR / "sales.parquet")
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    
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
        projected_growth = q4_last / (q4_prev + 1e-6)
        projected_median = train_medians[train_yr_max] * projected_growth
        
        # --- SCENARIO 1: Yearly Normalization (Current) ---
        def run_yearly():
            tr = train_raw.copy()
            tr['y'] = tr['Revenue'] / tr['year'].map(train_medians)
            tr['rev_lag_1_norm'] = (tr['Revenue'].shift(1).fillna(method='bfill') / tr['year'].map(train_medians))
            tr = tr.dropna()
            
            model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            model.fit(tr[['month', 'rev_lag_1_norm']], tr['y'])
            
            te = test_raw.copy()
            te['rev_lag_1_norm'] = (te['Revenue'].shift(1).fillna(tr['Revenue'].iloc[-1]) / projected_median)
            preds = model.predict(te[['month', 'rev_lag_1_norm']]) * projected_median
            return mean_absolute_error(test_raw['Revenue'], preds)

        # --- SCENARIO 2: Rolling Normalization (Proposed) ---
        def run_rolling():
            tr = train_raw.copy()
            # Use 364d rolling median for stationarity
            rolling_median = tr['Revenue'].rolling(364, min_periods=30).median().fillna(method='bfill')
            tr['y'] = tr['Revenue'] / rolling_median
            tr['rev_lag_1_norm'] = (tr['Revenue'].shift(1).fillna(method='bfill') / rolling_median)
            tr = tr.dropna()
            
            model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            model.fit(tr[['month', 'rev_lag_1_norm']], tr['y'])
            
            te = test_raw.copy()
            # For test, we use the last available rolling median * projected growth
            last_rolling = rolling_median.iloc[-1]
            projected_rolling = last_rolling * projected_growth
            
            te['rev_lag_1_norm'] = (te['Revenue'].shift(1).fillna(tr['Revenue'].iloc[-1]) / projected_rolling)
            preds = model.predict(te[['month', 'rev_lag_1_norm']]) * projected_rolling
            return mean_absolute_error(test_raw['Revenue'], preds)

        mae_yearly = run_yearly()
        mae_rolling = run_rolling()
        
        results.append({'fold': test_yr, 'yearly': mae_yearly, 'rolling': mae_rolling})
        
    res_df = pd.DataFrame(results)
    avg_yearly = res_df['yearly'].mean()
    avg_rolling = res_df['rolling'].mean()
    
    print("--- Normalization Strategy Verification (3-Fold, No Leak) ---")
    print(f"Base MAE (Yearly): {avg_yearly:,.0f}")
    print(f"Rolling MAE (364d): {avg_rolling:,.0f}")
    print(f"Gain: {avg_yearly - avg_rolling:,.0f} ({(1 - avg_rolling/avg_yearly)*100:.2f}%)")

if __name__ == "__main__":
    verify_rolling_norm()
